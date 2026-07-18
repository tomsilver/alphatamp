# DD2D — Drawer Decluttering in 2D (`blocks_tamp/dd2d/`)

DD2D is the fast, controlled 2D instrument of [`dd2d_spec.md`](dd2d_spec.md). A top-down
household **drawer** holds 9–14 rotated items; one is a **target** that starts
*ungraspable* (neighbours block every two-finger grasp). The robot stages a **subset of
blocker items** onto an adjacent **buffer** (a wall-less counter strip) to clear the
target, then retrieves it. The hard decision is *which subset to stage*: the staged set
must **jointly pack** into the limited buffer **and** be stageable/extractable by the
actual gripper. Feasibility hinges on a global, high-interaction-order continuous packing
statistic — the plan-feasibility signal (PIGINet/LAZY) this repo studies, in a form a
low-order classifier should struggle with.

DD2D is a **pivot from E1**, but a different world model (rotated parametric polygons + a
supporting-line grasp model + target-retrieval, vs. E1's axis-aligned shelf-loading), so
it is its own subpackage. **E1 stays intact.** DD2D reuses only the domain-agnostic layer
of `blocks_tamp` (`skeleton`, `record`/`PIGINetExample`, `RefineResult`/`BoundStep`,
`RenderResult`/`GeometryBackend`, `confirm_rendering`, `ObjectInfo`). Units are centimetres.

## Module map

| File | Role |
|---|---|
| `dd2d/shapes.py` | §4 parametric families (can/bowl/box/pillcase + concave dumbbell/shoe/banana) polygonised to **Shapely**, `concave` flag (fam 5–7), size sampling; every shape admits ≥1 isolation grasp (resample). |
| `dd2d/world.py` | `DrawerScene` (drawer + 1.5 cm wall band + buffer strip) + `DrawerWorld` (mutable occupancy, region-local collision) + `sample_buffer_pose` — the §6.3 **compaction-biased** sampler (contact/slide proposals + bottom-left push), shared by the refiner and the labeler. |
| `dd2d/grasps.py` | §5.3 supporting-line grasp model: `grasp_cells` (18 dirs × 5 slides, aperture 0.5–12 cm, contact-overlap interval), `finger_rects`, `grasp_cfree`, `has_grasp`. |
| `dd2d/scene.py` | §9.1 forward generator: sample drawer/buffer/fill, central target, settled-clutter loop. |
| `dd2d/enumerate.py` | §7 geometric candidate enumeration: target grasp cells → blocker sets → minimal sets → seeded supersets (≤40); clearing + memoised extraction-order re-checks. |
| `dd2d/label.py` | Day-1 labeler (see below) + the decision-relevance filters F1/F2/F3 (§9.4). |
| `dd2d/problem.py` | `DD2DProblem` (duck-types the record surface) + `generate_dd2d_problem` (generate → enumerate → label → filter → optional certification). |
| `dd2d/refine.py` | `DD2DRefiner` — the §10.2 backjumping refiner over `pick`/`place-buffer`/`retrieve`. |
| `dd2d/planning.py` | `DD2DPlanner` — candidate → staging skeleton, orderings `published`/`random`/`slack`/`oracle`. |
| `dd2d/render.py` | matplotlib polygon `render_scene` (→ `RenderResult`) + `render_episode` (elevated-carry video). Decoupled from the PyBullet stack. |
| `dd2d/demo.py` | `python -m blocks_tamp.dd2d.demo` — the end-to-end demo. |
| `domain/drawer_declutter.pddl` | geometry-blind STRIPS domain (`pick`/`place-buffer`/`retrieve`). |

The **only** shared-code edit is registering `"dd2d"` in `blocks_tamp.problem.PROBLEM_GENERATORS`
(lazy import). The planner's `domain_pddl_path` seam already existed (added for E1).

## Why a candidate-enumeration planner (not generic top-k)

The clearing decision is *geometric* — which items block the target's grasp fingers — and
that constraint is deliberately dropped from the symbolic model (§6.1). The shortest
optimistic plan is literally `retrieve(target)` (§6.2); a goal-directed planner (SymK
top-k / pyperplan) then just grows plans that stage *arbitrary* items — the blocking
subsets never surface within k. So, exactly as PIGINet obtains obstacle-moving skeletons by
replanning-with-obstacle-removal, `DD2DPlanner` enumerates the clearing subsets up front
(`enumerate.py`) and turns each into a staging skeleton
`[pick(o); place-buffer(o) …] ++ retrieve(target)`. (`--planner symk|pyperplan` still runs
the generic planner over the DD2D domain, to demonstrate the geometry-blind layer.)

## Fair baselines (`--planner pyperplan`)

For research, the geometry-blind **pyperplan** planner (`ForbidLoopPlanner`, a *standard*
diverse planner) is the fair baseline against the geometry-informed `candidates` planner. By
default it is a **deep, k-driven** enumeration: `make_dd2d_planner(prefer="pyperplan")` sets
`length_slack=None`, so `.plan(problem, k)` returns the `k` globally-shortest diverse plans
**ascending-length**, reaching multi-object stagings (with the generic default `length_slack=2`
it would be capped at `shortest+2 = 3` = single-object stagings only, so it returned ~`1 +
n_blockers` plans and never a 2-object staging). Pass `--pyperplan-slack <int>` to cap length.

Because pyperplan is geometry-blind it enumerates *arbitrary* blocker subsets, so a
subset-required instance needs a **large `k`** to reach the feasible plan: on the order of
`n_blockers^subset` — ≈150 for a 2-subset, ≈1500 for a 3-subset. The demo reports the
baseline metric **first-feasible rank / N** per problem and `solved N/num within k`,
`mean first-feasible rank` for the batch. Measured (λ=0.6, crowd=10): pyperplan `--k 200`
enumerates 200 plans and solves a 2-subset problem at **rank 85/200**; a 3-subset needs
`--k 800` → **rank 691/800**; `candidates` solves the same problems at **mean rank ~2.7**.
So the standard planner *can* reach the feasible plan, but at combinatorially higher
refinement cost — the intended PIGINet-style finding (ranking pays off). BFS enumeration is
cheap; the cost is *refining* the `k` plans.

This mode is opt-in via `length_slack=None`; the shared `ForbidLoopPlanner` default
(`length_slack=2`) is unchanged, so the other envs (sorting/stacking/clutter/E1) behave
exactly as before.

## The Day-1 labeler (what is deferred and why it's honest)

Every candidate is labeled `feasible` / `infeasible` / `marginal`:

- **feasible** ⇔ an extraction order exists (§7b) **and** a positive **accessible δ-packing
  certificate** is found — the subset packs into the buffer with ≥ δ clearance *and* an
  insertion order where each item's grasp clears the already-staged items (§8.2). The
  packing search reuses the compaction sampler on δ/2-inflated shapes.
- **infeasible** ⇔ no extraction order, **or** the sound **H1 area bound** proves no packing
  can exist (Σ δ/2-deflated areas > buffer area).
- **marginal** ⇔ neither: `inaccessible` (packs but no grasp order clears) or `budget`
  (no packing found within the restart budget — **provisional**).

The spec's **arrangement-complete negative certificate** (NFP + arrangement vertices +
Lipschitz rotation grid, §8.4) is the schedule-critical hard item and is **deferred**;
without it we cannot *prove* most packing-infeasibilities, so they land in
`marginal(budget)` rather than a hard `infeasible`. This is the spec's own Day-1 fallback —
honest and disclosed. **No label-dependent research numbers** should be reported until the
complete checker replaces it.

## The feasibility signal (what the demo shows), and the stated prior

Per problem, `DD2DProblem.candidates` are labeled and ordered; the refiner (the demo's
binary label) yields three outcomes, all visualised:

- **success** — the staged subset packs and the target is retrieved;
- **pick-failure** — a chosen blocker is itself buried (drawer-side extraction infeasible);
- **buffer-overflow** — an extractable-but-too-large subset can't pack (`stuck@ place-buffer(o)`;
  backjumping thrashes then budget exhausts) — the joint packing infeasibility.

Consistent with the spec's **stated prior** (§1.1/M8): the shape library is
convex-majority (four of seven families), so at loose buffers the *smallest* clearing subset always packs
(first-feasible ≈ rank 0) and infeasibility is dominated by extraction. The packing
headroom is **thin** and appears only at tight λ. This is *measured, not installed*: the
generator produces naturalistic scenes and the difficulty is a property of the
distribution — a falsification-leaning result is a legitimate (and expected) outcome.

## Requiring a blocking SUBSET (the `--crowd` knob)

By default (`crowd=0`) DD2D is *naturalistic* but almost always lets a **single** object
removal clear the target — the measured base rate of "the minimum feasible clearing subset
is ≥2" is only **~5–10%**. That is structural: a grasp direction is free iff *both* opposite
finger corridors are clear, and `settle_pose` drops each clutter item from a **random**
bearing, so a lone neighbour usually sits alone in one corridor → a size-1 minimal clearing
set that is itself feasible. Lowering λ does **not** help (it only grows the size-3
*candidate* tail; the *minimum feasible* subset stays 1).

To make problems **require identifying a 2+ blocker subset**, the target must be **pincered**:
every admissible grasp direction straddled by ≥2 items, so no single removal opens a corridor
and the required clearing set becomes a diametric **pair** (or larger). The `--crowd N` knob
(a disclosed placement prior, like clutter's Poisson radius / E1's tightness) does this: it
places a small-can **collar** of `N` items sliding *toward* the target from evenly-spaced
opposing bearings (`world.collar_pose`), and biases the target to a compact round shape so a
tight ring has few gaps while the collar stays graspable + packable. The
enumerator/labeler/refiner are unchanged — they already handle any-size subsets.

- **Natural mix (default):** `--crowd 10` (λ=0.6) makes **~50%** of problems require a 2+
  subset, the rest single-object. The demo prints the achieved fraction
  (`X/Y problems REQUIRED a 2+ blocker subset`). This is *measured, not installed*.
- **Hard guarantee:** `--min-subset N` keeps only instances whose smallest feasible clearing
  subset is ≥ N (~100%; cheap because crowding already raised the base rate). It **implies**
  `--require-subset`; `--require-subset` on its own is the same as `--min-subset 2`. Off by default.
- **See the subsets:** use the **default `candidates` planner** — `--planner pyperplan`/`symk`
  are geometry-blind and within small `k` only ever propose single-object stagings, masking
  subset-required scenes.
- **Diverse collar (`--diverse-crowd`):** by default the collar is round cans only, so concave
  shapes (dumbbell/shoe/banana) only reach the outer clutter and act as distractors, never
  entering a feasible plan. `--diverse-crowd` draws the collar from **all** families so concave
  shapes join the pincer. The ring is looser (non-round items leave larger angular gaps and fail
  `collar_pose` more often), so the natural subset rate drops — measured **~50% → ~10%** at
  `--crowd 10` (λ=0.6); pair with `--require-subset` to restore ~100%. Datasets get a `dc`
  problem-id marker. Caveat: `--min-subset 3 --diverse-crowd` is often too rare and can exhaust
  `max_resamples` (a loud `RuntimeError`); raise `--crowd`, lower the floor, or drop `--diverse-crowd`.

A subset-required problem's feasible plan is e.g.
`pick(C);place-buffer(C);pick(C′);place-buffer(C′);retrieve(target)` (2 objects); staging only
one of them then retrieving is infeasible, stuck at `retrieve(target)` — one object isn't
enough. `min_feasible_subset` / `requires_subset` are stored on `DD2DProblem`.

Tuning note: with equal target/blocker finger sizes there is a real tension — a gap tight
enough to block the target also makes collar items un-graspable — so a *pure* natural rate
above ~50% is hard; the compact-target + small-collar combo and `--require-subset` are the
levers. `crowd ≈ 10` is the sweet spot (crowd 8 → ~35%, 10 → ~50%, 12 → ~44% as the ring
starts burying itself).

## Knobs

- **`--crowd`** — collar crowding (subset-difficulty dial): 0 = naturalistic (~5% require a
  subset), 10 = ~50/50 mix (see above). `--require-subset` / `--min-subset N` guarantee it.
- **λ (`--lambda`)** — buffer scale (spec P4). Smaller = tighter buffer = more
  buffer-overflow (joint packing) failures; the interesting regime is ~[0.75, 0.9].
- **δ (`--margin`)** — label margin / refiner slack (spec P12); larger = easier packing.
- **`--order`** — candidate ordering (`published`/`random`/`slack`/`oracle`).
- **`--num-items`** — items incl. target (default sampled 9–14); keep ≤ ~12 if using
  `--planner symk` (SymK's BDD wall).

### Refiner budget knobs (spec P13/P14/P15)

The `DD2DRefiner` cost model is tunable on the demo/generator (the DD2D data-gen path), so a
demo or a data-collection run can dial refinement effort. Three levels, plus a wall-clock cap:

- **`--max-stream-calls`** (B, total stream calls, P13; default 300) — the global cap.
  `--max-stream-calls 0` (or `<=0`) **disables** it, so refinement is governed by time + per-step
  attempts instead.
- **`--samples-per-step`** (`m_p`, P15; default 15) — candidate poses tried *inside* one
  `sample-buffer-pose` call (the best-packed is kept). This is the **sampler-adequacy dial**:
  raise it if feasible subsets spuriously fail to pack at tight λ (spec §11(a)). Costs EGEs, not
  stream calls.
- **`--retry-cap`** (`t`, P14; default 10) — `sample-buffer-pose` *calls* per `place-buffer` step
  before backjumping. Secondary (backjump-timing) knob; default is usually fine.
- **`--time-budget`** (seconds/plan; default unbounded) — stops refinement when wall-clock is hit.
  Combine with `--max-stream-calls 0` to run **purely by time + per-step**.

Refinement stops at the stream-call cap **or** the wall-clock cap (whichever first); if both are
disabled the refiner warns and falls back to `budget=300`. These knobs are threaded into
**generation-time certification**, so every kept problem has ≥1 plan that refines under your
budget (a tighter budget just resamples more). The three-way **labeler** (`label.py`, used for
F1/F2/F3 + `min_feasible_subset`) deliberately stays at a fixed generous `m_p=15` — ground-truth
problem selection must not depend on the collection budget; only the refiner (whose result is the
collected per-plan label) is tunable.

## Commands

```shell
.venv/bin/python -m pip install -r blocks_tamp/requirements-blocks.txt   # shapely already pinned
.venv/bin/python -m blocks_tamp.dd2d.demo --lambda 0.6 --seed 0 --num-problems 10 --crowd 10   # ~50% require a 2+ subset
.venv/bin/python -m blocks_tamp.dd2d.demo --lambda 0.6 --seed 0 --num-problems 5 --require-subset  # guarantee a subset
.venv/bin/python -m blocks_tamp.dd2d.demo --lambda 0.55 --seed 2 --crowd 0 --order slack  # naturalistic baseline
.venv/bin/python -m pytest blocks_tamp/tests/test_dd2d.py -q
```

Outputs land in `out_dd2d/`: one `PIGINetExample` JSON per (problem, plan), a top-down
execution video per selected plan (always incl. the first feasible — `success_*` full
retrieval, `failure_*` partial-to-failure with red overflow ghosts + verdict banner), and a
`render_check_*.png` confirmation frame.

## Deferred (the world/label/record layer is structured to carry them)

Tier-1 off-the-shelf PDDLStream + FastDownward baselines (§10.1); the 7-variant two-tier
protocol beyond the candidate orderings shipped (§10.2); attack suites — heuristic
certificates H2–H4 + Tier-0 learned models, `scikit-learn` (§10.4); the buffer-slack λ
sweep with bootstrap CIs (§11); the arrangement-complete negative certificate (§8.4); the
§9.5 filter-shift and §10.3 coverage audits; the held-out-generator split (§4).
