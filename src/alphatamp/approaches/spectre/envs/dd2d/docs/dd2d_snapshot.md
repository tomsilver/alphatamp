# DD2D — implementation snapshot (2026-07-08)

A **standalone, current-state reference** for DD2D (Drawer Decluttering in 2D) as implemented in
`blocks_tamp/dd2d/`. It describes the objects, the generator, the planners / refiner / heuristics, the
data-and-record layer, the tooling, the tests, and what is still deferred — enough to understand or run
the system without reading the source.

This is a *snapshot*, not the design rationale. For the "why," see [`dd2d.md`](dd2d.md); for the full
spec, [`dd2d_spec.md`](dd2d_spec.md); for the CLI-argument tutorial, [`dd2d_demo_args_faq.md`](dd2d_demo_args_faq.md).
Units are **centimetres** throughout.

> **Note vs. older docs:** the shape library was fully replaced (7 families now, three concave), a
> geometric **A\*/GBF distance-heuristic** search path was added, and a **PIGINet data pipeline**
> (`record_ext.py`, `collect.py`, `inspect_example.py`, `heuristic_experiment.py`) landed. Anything in
> older notes referencing families like `bottle/board/mug/spray/spatula/lid` is out of date.

---

## 1. What DD2D is

A top-down 2D **household drawer** holds 9–14 rotated rigid items; one is the **target**, which starts
*ungraspable* — its neighbours block every two-finger grasp. The robot uses a top-down parallel-jaw
gripper. To retrieve the target it must **stage a subset of blocker items onto an adjacent buffer** (a
wall-less counter strip) to clear a grasp corridor, then grab the target. The hard decision each episode
is **which subset of blockers to stage**: the staged set must **jointly pack** into the limited buffer
*and* be stageable/extractable by the actual gripper.

DD2D is the fast 2D instrument for the repo's research question: whether skeleton feasibility can hinge
on a **global, high-interaction-order, continuous packing statistic** (which subset fits) that a
low-order feasibility classifier (PIGINet-style) should struggle to predict — and at what buffer
tightness that gap appears (or doesn't).

---

## 2. Module map (`blocks_tamp/dd2d/`)

| File | Role |
|---|---|
| `shapes.py` | Parametric shape library (7 families), `sample_shape`, isolation-graspability, holdout split. |
| `grasps.py` | Top-down parallel-jaw grasp model (`grasp_cells`, `has_grasp`, `finger_rects`, `isolation_graspable`). |
| `world.py` | `DrawerScene` / `DrawerWorld` / `ItemState` / `StreamCounter`; the compaction buffer sampler `sample_buffer_pose`, `settle_pose`, `collar_pose`. |
| `scene.py` | Forward scene generator `generate_scene` (drawer/buffer geometry, target, collar, settled clutter). |
| `enumerate.py` | Geometric candidate enumeration (blocker sets → minimal sets → supersets → clearing + extraction re-checks). |
| `label.py` | Three-valued labeler (`feasible`/`infeasible`/`marginal`) + decision filters F1/F2/F3. |
| `problem.py` | `DD2DProblem` + `generate_dd2d_problem` (generate → enumerate → label → filter → certify). |
| `refine.py` | `DD2DRefiner` — the shared backjumping refiner over `pick`/`place-buffer`/`retrieve`. |
| `planning.py` | `DD2DPlanner` (candidate enumerator) + `make_dd2d_planner` (candidates / pyperplan / symk, bfs/gbf/astar, heuristic wiring). |
| `heuristics.py` | Hand-written geometric **distance heuristic** (`distance_heuristic_factory`, forms `inv`/`avg`/`radius`). |
| `render.py` | matplotlib `render_scene` + `render_episode` (elevated-carry video) + `DD2DRenderBackend`. |
| `demo.py` | End-to-end demo CLI (generate → plan → refine → records + videos). |
| `eda.py` | Geometry-blind pyperplan **difficulty EDA** (attempts-until-success + success prob). |
| `collect.py` | DD2D-native **PIGINet dataset collector** (balanced strata, parallel). |
| `record_ext.py` | Geometry sidecar over the shared `record.PIGINetExample` (poses, shapes, crops). |
| `inspect_example.py` | Single-record visualizer → a PIGINet-input figure. |
| `heuristic_experiment.py` | 5-arm (bfs / astar-hff / gbf-hff / astar-dist / gbf-dist) first-feasible-rank comparison. |
| `heuristic_notebook.py` | marimo dashboard for the heuristic experiment. |
| `eda_notebook.py` | marimo dashboard for the EDA. |
| `render_families.py` | Renders a labelled gallery PNG of the shape families. |
| `../domain/drawer_declutter.pddl` | Geometry-blind STRIPS domain (3 actions). |
| `__init__.py` | Public exports (matplotlib-free: `render` is intentionally *not* imported). |

`__init__.py` exports: `Shape, sample_shape, Grasp, grasp_cells, finger_rects, grasp_cfree, has_grasp,
isolation_graspable, ItemState, DrawerWorld, place_polygon, sample_buffer_pose, DrawerScene, DD2DProblem,
generate_dd2d_problem, make_dd2d_problem, DD2DRefiner, DD2DPlanner, make_dd2d_planner`. The heavier
tooling (`collect`, `record_ext`, `inspect_example`, the notebooks) is run as `python -m
blocks_tamp.dd2d.<module>`.

---

## 3. Objects — the shape library (`shapes.py`)

**7 parametric families.** Each sampled item draws a family + dimensions, is polygonised to a Shapely
polygon (curved shapes ~24–28 vertices), and is recentred so its centroid is at the origin (a pose
`(x, y, θ)` rotates about the centroid then translates).

| Family | Convex? | Dimension ranges (cm, before holdout shift) | Shape |
|---|---|---|---|
| `can` | convex | diameter `U(4, 8)` | small–medium circle |
| `bowl` | convex | diameter `U(8, 12)` | large circle (capped near the 12 cm aperture) |
| `box` | convex | w `U(5, 20)` × h `U(4, 12)`; 50% sharp / 50% rounded (radius `U(0.3, 1.0)`) | rectangle |
| `pillcase` | convex | length `U(10, 18)` × width `U(2, 4)` | capsule (rect + two semicircular caps) |
| **`dumbbell`** | **concave** | ends `U(3,5)×U(4,7)`, bar_len `U(4,8)`, bar_t `U(1.5,2.5)` | **waist** (two end blocks + thin bar) |
| **`shoe`** | **concave** | thickness `U(3,5)`, arms `U(7,11)` each | **L-corner** (two rects) |
| **`banana`** | **concave** | r_out `U(5,7)`, thick `U(2,3)`, opening half-angle `U(55°,75°)` | **C-opening** (single simple polygon) |

Notes:
- **Concave set** = `{dumbbell, shoe, banana}` (the `concave` flag). These are the "tricky convexity"
  shapes. They are built to **always be a single valid polygon**: `dumbbell`/`shoe` are Shapely
  `unary_union`s with a small `_OVERLAP = 0.4` cm so shared edges merge; `banana` is constructed directly
  as one simple polygon (outer arc out, inner arc back, gap on the +x side — **no annulus hole**, so
  validity is guaranteed).
- **Sampling weights** (`_FAMILY_WEIGHTS`): `can` = 1.3, `box` = 1.3, all others = 1.0.
- **Isolation-graspability:** every sampled shape must admit ≥1 grasp *in isolation* (some direction with
  width ≤ 12 cm aperture and a non-empty contact-overlap interval); non-graspable draws are resampled
  (`sample_shape(..., require_graspable=True)`, up to `max_tries=40`).
- **`Shape` dataclass** (frozen): `family`, `polygon` (centroid at origin), `concave`; properties `size`
  (bbox w,h), `area`, `r_max` (max centroid-to-boundary distance).
- **`split` / holdout:** `sample_shape(rng, family=None, split="train")`. `split="holdout"` scales every
  dimension band by **×1.15** and swaps `bowl → can`. (Wired but not yet used in a formal train/test
  experiment.)
- Preview the families: `python -m blocks_tamp.dd2d.render_families` → `out/dd2d/shape_families.png`.

---

## 4. Grasp model (`grasps.py`)

A **top-down parallel-jaw grasp** = two finger rectangles pressed flush against opposite sides of an
item, both collision-free.

- Constants: `FINGER_WIDTH = 2.5` (tangential), `FINGER_THICK = 2.0` (normal), aperture `[MIN 0.5, MAX
  12.0]`, `N_DIRECTIONS = 18` (10° steps over `[0, 180)`), `N_SLIDES = 5`, interior fraction `0.80`.
- `direction_admissible(shape, α)` → whether a grasp along `α` fits the aperture and has a non-empty
  contact-overlap interval; `grasp_cells(shape)` enumerates all admissible `(α, slide)` cells (no
  collision filtering); `finger_rects(g, pose)` gives the world-frame finger rectangles; `grasp_cfree(g,
  pose, obstacles)` tests them against obstacles; `has_grasp(shape, pose, obstacles)` returns the first
  collision-free grasp cell (or `None`); `isolation_graspable(shape)` is the resample gate used by the
  shape sampler.

Height stratification is analytic (no z simulated): items are 6 cm prisms, walls 12 cm, fingers descend
to grasp depth — so every manipulation phase reduces to a 2D collision query.

---

## 5. World model (`world.py`)

- **`DrawerScene`**: `drawer` (Polygon), `wall_band` (1.5 cm ring around the interior — blocks fingers),
  `buffer` (the staging strip), `items: dict[str, ItemState]`, `target`, `margin` (δ), `dims`. Helpers
  `item_names()`, `blockers()`, `target_state()`.
- **`ItemState`**: `name`, `shape`, `pose (x,y,θ)`, `region` ∈ {`drawer`,`buffer`,`hand`,`removed`},
  `is_target`; `footprint()` → placed polygon.
- **`DrawerWorld`**: mutable occupancy for the refiner — `pick`, `place_buffer`, `extract`,
  `snapshot`/`restore`, `drawer_obstacles` (includes the wall band), `buffer_obstacles` (no walls).
- **`StreamCounter`**: tracks `calls` and `eges` (elementary geometric evaluations) — the spec's two cost
  units.
- **`sample_buffer_pose(shape, buffer_poly, staged_polys, rng, m_p=15, inflate=0.0, beta=0.3)`** — the
  **compaction-biased** sampler: draws `m_p` candidate poses on a 15° orientation grid (0.7 contact /
  0.3 slide proposals), pushes each bottom-left, scores `cx + 0.01·cy + gumbel(β)`, returns the lowest.
  `inflate = δ/2` when the **labeler** certifies a packing; `inflate = 0` for the **refiner** (real
  geometry). This is the single component shared by the refiner and the labeler.
- `settle_pose` (settled-clutter placement) and `collar_pose` (the crowd pincer prior) are also here.

---

## 6. The generator

### 6.1 Scene geometry (`scene.py`)

Sampled per instance (spec P1/P3/P5/P6):

| Constant | Value | Meaning |
|---|---|---|
| `DRAWER_W` | `U(35, 50)` cm | drawer interior width |
| `DRAWER_D` | `U(28, 40)` cm | drawer interior depth |
| `BUFFER_L` | `U(25, 45) × λ` cm | buffer length |
| `BUFFER_D` | `U(12, 20) × λ` cm | buffer depth |
| `WALL_BAND` | `1.5` cm | wall ring (blocks fingers) |
| `BUFFER_GAP` | `6` cm | buffer sits 6 cm right of the drawer |
| `FILL_RANGE` | `(0.35, 0.55)` | target fill fraction (Σ item area / drawer area) |
| `N_RANGE` | `(9, 14)` | item count incl. target |
| `BOX_MAX_FRAC` | `0.45` | reject a `box` whose bbox area > 45% of drawer short-side² |

**λ (the difficulty dial)** scales the buffer's *own* nominal size, not the drawer — so the buffer is a
narrow counter strip that is smaller than the drawer at all normal λ (≈17% of drawer area at λ=0.6, ≈47%
at λ=1.0). This keeps buffer capacity **below full-evacuation**, which is what forces a subset choice.
Interesting regime ≈ **[0.75, 0.9]**. See [`dd2d_demo_args_faq.md`](dd2d_demo_args_faq.md) for the full
λ-vs-buffer discussion.

`generate_scene(seed, lam=1.0, split="train", fill=None, n_items=None, crowd=0, diverse_crowd=False)`:
1. Sample drawer/buffer geometry, `fill`, `n_items`.
2. **Target**: placed uniformly over the central 50%×50%, any rotation; biased to a compact `can` when
   `crowd > 0` (so a tight collar can ring it).
3. **Collar / crowd** (if `crowd > 0`): ring the target from `crowd` evenly-spaced bearings so opposing
   items straddle grasp corridors → clearing needs a diametric **pair** (2+ subset). Collar family is
   `can` only (`_COLLAR_FAMILIES = ("can",)`) unless `diverse_crowd=True`, in which case the collar is
   drawn from all families (concave shapes can join the pincer).
4. **Settled clutter**: fill with items (any family, `settle_pose`) until `len(items) == n_items` or the
   fill fraction is reached.

> **Collar counts within `n_items`:** the collar items are part of the `n_items` budget, not added on
> top — a `--crowd 10` scene still holds ~9–14 items, not ~19–24.

### 6.2 Problem assembly + labeling (`problem.py`)

```python
generate_dd2d_problem(
    lam=1.0, seed=0, margin=1.0, split="train",
    n_items=None, crowd=0, diverse_crowd=False,
    require_subset=False, min_subset=2, unblocked_target=False,
    certify=True, budget=300, retry_cap=10, samples_per_step=15,
    time_budget=None, max_resamples=400,
) -> DD2DProblem
```

Pipeline, resampled (new `scene_seed`) until it passes, up to `max_resamples`:
1. `generate_scene(...)`; set `scene.margin = margin`.
2. `enumerate_candidates(scene)` → geometric clearing candidates (§7).
3. `label_all(scene, candidates)` → three-valued labels (§8).
4. `decision_filters` → **F1** target actually blocked, **F2** ≥2 distinct clearing subsets, **F3** ≥1
   confidently-feasible candidate.
5. Branch:
   - **`unblocked_target=True`** (stratum 0): keep only if the target is *open* (`not F1`); set
     `min_feasible_subset = 0`, intended = the retrieve-only plan.
   - else require `F1 ∧ F2 ∧ F3`, and optionally **F4** (`require_subset` / `min_subset`): keep only if
     the smallest feasible clearing subset ≥ `min_subset`.
6. **Certification** (`certify=True`): the intended staging skeleton must refine under
   `(budget, retry_cap, samples_per_step, time_budget)`, **and** the degenerate `retrieve(target)`-only
   skeleton must *not* refine (target stays blocked). Guarantees each kept problem is solvable under the
   collection budget by construction.

**Strata** = `min_feasible_subset` ∈ **{0, 1, 2, 3}**: 0 = open target (retrieve-only works), 1 = a
single removal clears it, 2/3 = a diametric pair / triple must be staged. `requires_subset` ⇔
`min_feasible_subset ≥ 2`.

**`DD2DProblem`** duck-types the record surface: `problem_id` (`dd2d_n{n}_l{λ×100}[_c{crowd}[dc]]_s{seed}`),
`objects`, `tables=["drawer","buffer"]`, `init_facts`, `goal_facts`, `scene`, `seed`, `num_blocks`,
`num_blockers`, `target`, `candidates`, `lam`, `margin`, `crowd`, `diverse_crowd`, `min_feasible_subset`,
`intended`. Methods: `requires_subset`, `feasible_candidates()`, `intended_skeleton()`,
`retrieve_only_skeleton()`, `to_pddl_problem()`, `domain_pddl_path`.

---

## 7. Candidate enumeration (`enumerate.py`)

The clearing decision is geometric, so a dedicated enumerator (not generic top-k) surfaces the blocking
subsets. `enumerate_candidates(scene, seed)`:
1. For each usable target grasp cell, collect the items whose footprints hit its fingers → **blocker
   sets** (`_blocker_sets`); walled-off cells are discarded.
2. Reduce to inclusion-**minimal** sets (`_minimal_sets`).
3. Grow **supersets**: each minimal set ∪ one adjacent item (`ADJACENCY = 2.0` cm, seeded), up to
   `MAX_CANDIDATES = 40`.
4. Per candidate, two re-checks: **clearing** (with the subset removed the target has ≥1 collision-free
   grasp) and **extraction order** (a memoised DFS finding a per-member removal order where each member
   has a clear grasp; retained-but-flagged if none).

Each **`Candidate`** = `subset` (frozenset), `members` (removal order), `extractable`,
`extraction_reason`, `meta` (filled by the labeler), and `.size`. Published order = ascending `|S|`,
seeded ties. A candidate ≡ a staging skeleton `[pick(o); place-buffer(o) …] ++ retrieve(target)`.

---

## 8. Labeler (`label.py`) — and the honesty caveat

Every candidate is labeled **`feasible` / `infeasible` / `marginal`**:
- **feasible** ⇔ an extraction order exists **and** a positive **accessible δ-packing certificate** is
  found — the subset packs into the buffer with ≥ δ clearance *and* there is an insertion order where
  each item's grasp clears the already-staged items (packing search reuses the compaction sampler on
  δ/2-inflated shapes; `RESTARTS = 3`).
- **infeasible** ⇔ no extraction order, **or** the sound **H1 area bound** proves no packing can exist
  (Σ of δ/2-deflated areas > buffer area — one-directional, infeasibility only).
- **marginal** ⇔ neither: `inaccessible` (packs but no grasp order clears) or `budget` (no packing found
  within the restart budget — **provisional**).

Helpers: `min_feasible_subset_size(candidates)` (smallest `feasible` subset size, `None` if none),
`decision_filters(scene, candidates)` (F1/F2/F3).

> **Honesty gate:** the spec's **arrangement-complete negative certificate** (§8.4) is **deferred**.
> Without it, most packing-infeasibilities land in `marginal(budget)` rather than a hard `infeasible`.
> **No label-dependent research numbers should be reported** until the complete checker replaces this
> Day-1 fallback. (Disclosed, per spec.)

---

## 9. Refiner (`refine.py`)

A single shared **`DD2DRefiner`** (`name = "dd2d-backjump"`, `label_source = "refine_buffer_stage"`):
sequential binding with backjumping over a staging skeleton, replayed against a `DrawerWorld`.

```python
DD2DRefiner(budget=300, retry_cap=10, samples_per_step=15, time_budget=None)
```
- **`budget`** (B) — total stream-call cap; `≤0`/`None` = uncapped.
- **`retry_cap`** (t) — `sample-buffer-pose` calls per `place-buffer` step before backjumping.
- **`samples_per_step`** (m_p) — candidate poses tried inside one `sample_buffer_pose` call.
- **`time_budget`** — wall-clock seconds per plan.
- If both `budget` and `time_budget` are disabled → warns and falls back to `budget=300` (safety
  backstop `_SAFETY_CALLS = 5_000_000`).

Step semantics: `pick(o)` needs a grasp clearing remaining drawer items + wall (else hard dead-end);
`place-buffer(o)` samples up to `retry_cap` collision-free *and* graspable poses (accessibility), else
backjumps to re-sample the previous placement; `retrieve(target)` needs the target grasp to clear.
`refine(skeleton, scene, seed)` returns a shared **`RefineResult`**: `status`
(`feasible`/`infeasible`), `steps_bound`, `plan_length`, `n_attempts` (stream calls), `failure_action`,
`bound_plan`, `elapsed`.

**Three outcomes** (the demo's feasibility signal): **success** (subset packs, target retrieved);
**pick-failure** (a chosen blocker is itself buried — drawer-side extraction infeasible); **buffer-overflow**
(extractable but too-large subset can't pack — `stuck@ place-buffer(o)`, the joint-packing infeasibility).

---

## 10. Planners (`planning.py` + generic `blocks_tamp/planning.py`)

Three planner families, selected via `make_dd2d_planner(prefer=…, order=…, search=…, heuristic=…)`:

**(a) `prefer="candidates"` (default) — geometry-informed.** `DD2DPlanner` turns each enumerated clearing
candidate into a staging skeleton. Orderings (`order=`): `published` (ascending `|S|`), `random` (seeded
shuffle), `slack` (ascending `Σ area(S) / buffer_area`, ties by size — the strongest cheap ordering),
`oracle` (feasible-labeled first). Ranks the feasible plan near the top (mean rank ≈ 2–3).

**(b) `prefer="pyperplan"` / `"symk"`, `search="bfs"` — geometry-blind fair baseline.** The generic
`ForbidLoopPlanner` with `length_slack=None` (unbounded, k-driven) enumerates the k globally-shortest
diverse plans ascending-length. Because it's geometry-blind it stages arbitrary subsets, so a
subset-required instance's first feasible plan sits at a rank ~`n_blockers^subset` (≈85/200 for a
2-subset; ≈691/800 for a 3-subset). This cost gap vs. `candidates` is the intended PIGINet-style finding
(ranking pays off).

**(c) A\* / GBF with a distance heuristic — the newest addition.** *Not a new planner class*: it is
`search="astar"` or `"gbf"` on the same `ForbidLoopPlanner`, with priority `g + h` (astar) or `h` alone
(gbf). The heuristic is resolved by `_resolve_heuristic(name)`:
- `hff` / `hadd` — off-the-shelf pyperplan symbolic heuristics (geometry-blind).
- `dist` / `dist-avg` / `dist-radius` — the **new geometric distance prior** from `heuristics.py`.

Select it programmatically with `make_dd2d_planner(prefer="pyperplan", search="astar", heuristic="dist")`,
or on the demo with `--planner pyperplan --pyperplan-search astar --pyperplan-heuristic dist`.

`staging_skeleton(target, members)` builds `[pick(o); place-buffer(o) for o in members] ++
retrieve(target)`.

---

## 11. The geometric distance heuristic (`heuristics.py`)

`distance_heuristic_factory(form="inv", eps=1.0, radius_margin=2.0, use_edge=False)` returns a factory
`(task, problem) -> (node -> float)`. It scores the **proximity mass of non-target items still in the
drawer**, which *falls* as near (likely-blocking) items get staged — the correct sign to minimise for
gbf/astar. Forms:
- **`inv`** (default): `h = Σ 1/(dist + eps)` over remaining non-target in-drawer items — parameter-free.
- **`avg`**: `h = −mean(dist)` over remaining (negate so removing near items lowers h).
- **`radius`**: `h = Σ max(0, R − dist)` with `R = target.r_max + radius_margin` — closeness within a
  band.

Distances are item-centroid → target-centroid (or footprint-edge if `use_edge`). This is a **coarse
spatial prior**, deliberately *not* the exact grasp-blocker oracle in `enumerate.py` — it's the kind of
cheap geometric signal a learned model (PIGINet) might approximate.

---

## 12. PDDL domain (`../domain/drawer_declutter.pddl`)

A **geometry-blind STRIPS** domain — the collision/packing structure is deliberately dropped and
certified only in refinement. **3 actions only:** `pick`, `place-buffer`, `retrieve`. Predicates:
`in-drawer`, `on-buffer`, `holding`, `handempty`, `target`, `extracted`. The shortest optimistic plan is
literally `retrieve(target)` — "just grab it" — which fails when the target is blocked, after which the
planner grows longer staging plans.

> The spec discusses a legal `place-drawer` action; it is **not in the shipped domain or refiner** (which
> handle only `pick`/`place-buffer`/`retrieve`).

---

## 13. Data / record layer (`record_ext.py`)

DD2D feasibility is geometric, but the shared `record.PIGINetExample` schema is symbolic (constant init
literals, bbox-only sizes). `record_ext.py` adds the geometry channel **without changing the shared
schema** (its `objects` / `init_literals` are free-form lists):
- **`write_crops(problem, images_dir, views=("topdown",))`** — renders the scene once, writes one
  segmented crop PNG per object (`<obj>__<view>.png`) plus a full `scene.png`; reuses
  `record.build_image_refs` for seg-id/bbox.
- **`build_dd2d_example(problem, skeleton, refine_result, planner_name, images=None, …)`** — calls the
  base `record.build_example`, then augments each object with `pose: [x,y,θ]` and `shape: {family, w, h,
  area, concave}`, appends an `["at-pose", name, [x,y,θ]]` init literal per object, and records
  `provenance["drawer_wh"]` + `provenance["buffer_bounds"]` (a normalization reference). Rounded to 4
  decimals.

Collector-supplied provenance keys: `stratum`, `n_items`, `plan_idx`, `split`, `refine_seed`,
`planner_search`, `planner_heuristic`.

---

## 14. Tooling / CLIs

All run as `.venv/bin/python -m blocks_tamp.dd2d.<module>`. Key flags + defaults:

### `demo.py` — end-to-end demo (records + videos)
`--num-items` (None→9–14), `--lambda` (0.8), `--margin` (1.0), `--split {train,holdout}` (train), `--k`
(12), `--seed` (0), `--num-problems` (1), `--order {published,random,slack,oracle}` (published),
`--planner {candidates,symk,pyperplan}` (candidates), `--pyperplan-slack` (none),
**`--pyperplan-search {bfs,gbf,astar}` (bfs)**, **`--pyperplan-heuristic {hff,hadd,dist,dist-avg,dist-radius}` (hff)**,
`--crowd` (10), `--diverse-crowd` (flag), `--require-subset` (flag), `--min-subset` (None; **implies
`--require-subset`**; require-subset alone = floor 2), `--max-stream-calls` (300), `--retry-cap` (10),
`--samples-per-step` (15), `--time-budget` (None), `--no-certify` (flag), `--max-videos` (6),
`--video-format {mp4,gif}` (mp4), `--out-dir` (out_dd2d).

### `eda.py` — geometry-blind difficulty EDA
Always pyperplan (no `--planner`). Produces the **attempts-until-success** (first-feasible rank / excess
`E = rank−1`) distribution and **Wilson-CI success probability**, stratified by `min_feasible_subset ∈
{1,2,3}` and pooled. Flags: `--episodes` (200), `--workers` (8), `--k` (200), `--lambda` (0.8),
`--crowd` (10), `--max-stream-calls` (500), `--time-budget` (10.0), `--retry-cap` (10),
`--samples-per-step` (15), `--max-scan-seeds` (4000), `--calibrate` / `--smoke` / `--analyze-only`.
Outputs `out/dd2d_eda/{episodes.csv, summary.json}`; view with `eda_notebook.py` (marimo).

### `heuristic_experiment.py` — 5-arm rank comparison
Compares **bfs / astar-hff / gbf-hff / astar-dist / gbf-dist** on first-feasible rank over
subset-requiring problems (a feasibility label is memoised per skeleton with a stable seed, so it's
identical across arms). If symbolic `hff` doesn't help but geometric `dist` does, the useful signal is
geometric. Defaults: `--num-problems` (50), `--lambda` (0.8), `--crowd` (5), `--diverse-crowd` (on),
`--num-items` (13), `--min-subset` (3), `--k` (200), `--max-stream-calls` (500), `--time-budget` (10.0),
`--max-expansions` (200_000), `--max-scan-seeds` (6000), `--smoke` / `--analyze-only`, `-o/--output`.
Writes `out_dd2d/heuristic_experiment/results.csv` (+ meta JSON); dashboard `heuristic_notebook.py`.

### `collect.py` — DD2D-native PIGINet dataset collector
Balanced min-subset **strata (0,1,2,3)**, parallel, planned with **astar + dist**. Per problem:
generate a requested-stratum instance → plan k skeletons → refine and persist. **`full_pool=True`**
(default) refines *all k* plans (many positive+negative per problem, no length confound); legacy mode
stops at first feasible; drops the problem if unsolved within k. Disjoint seed bands per split and
per-stratum. CLI: `--out-root` (data/dd2d/raw), `--workers` (8), `--target-train` (400), `--target-test`
(100), `--target-val` (100), `--splits` (train,test,val), `--band` (1_000_000), `--crowd` (5),
`--lambda` (0.8), `--time-budget` (20.0), `--resume`, `--smoke`. Locked config: `k=200`, `budget=None`
(time-governed), `diverse_crowd=True`. Output: `<split>/<problem_id>/{images/, scene.png, NNN.json}` +
per-split `manifest.json` + `attempted.log`.

### `inspect_example.py` — single-record visualizer
Renders one figure mapping a saved `PIGINetExample` 1:1 to PIGINet inputs (initial state, plan, goal +
init literals, pose/shape value features, per-object crops, text vocab, label + refine diagnostics +
provenance). Reads only the record JSON + sibling `scene.png`/`images/`. CLI: positional `record` (a
json file, or a problem/split/dataset dir — auto-picks a record), `--out` (default
`<record_dir>/inspect.png`).

### `render_families.py` — shape gallery
`--out` (out/dd2d/shape_families.png), `--samples` (3), `--seed` (1000).

---

## 15. Rendering (`render.py`)

matplotlib/PIL/imageio, decoupled from PyBullet.
- `render_scene(scene, width=720, view="topdown", poses=None) -> RenderResult` — rgb + per-item
  segmentation + id→name; `poses` overrides item poses to render arbitrary states. Consumed unchanged by
  `record.build_image_refs` and `rendering.confirm_rendering`.
- `render_episode(scene, bound_plan, feasible, failure_action, out_path, fmt="mp4", fps=20)` — replays a
  bound plan with the **elevated-carry** convention (carried item drawn as a dashed no-fill outline +
  drop shadow), rejected buffer poses flash as red ghosts, and an infeasible plan runs its bound prefix
  then draws the failing action's overflow ghost / blocked fingers + a verdict banner. mp4 needs
  `imageio-ffmpeg` (else gif).
- `DD2DRenderBackend` (`name="dd2d-matplotlib"`) adapts `render_scene` to the shared `GeometryBackend`
  ABC.

---

## 16. Tests

| File | Count | Coverage |
|---|---|---|
| `tests/test_dd2d.py` | 46 | shapes/grasps, buffer sampler + world, scene/generation, crowd/collar/subset, candidates/labels, planners (incl. the gbf/astar heuristic arms), refiner, record/render. |
| `tests/test_dd2d_collect.py` | 12 | full-pool vs legacy collection, drop-unsolvable, exact-stratum rejection, determinism, band disjointness, manifest/disk layout, resume. |
| `tests/test_dd2d_record_ext.py` | 2 | crop-per-object PNGs, geometry-augmented example round-trip. |
| `tests/test_dd2d_inspect.py` | 2 | record→figure, record resolution from a split dir. |

Run all DD2D tests: `.venv/bin/python -m pytest blocks_tamp/tests/test_dd2d*.py -q`.

---

## 17. Status — implemented vs. deferred

**Implemented (this snapshot):** the world/grasp/label/record/render layer; the forward generator with
crowd/diverse-crowd/unblocked-target strata and the F1–F4 filters + certification; candidate enumeration;
the three-valued Day-1 labeler; the shared backjumping refiner; **all three planner families** including
the new geometric **A\*/GBF distance-heuristic** path; the geometry-blind difficulty **EDA**; the 5-arm
**heuristic experiment**; the DD2D-native **dataset collector**, geometry **record sidecar**, and record
**inspector**.

**Deferred** (see [`dd2d.md`](dd2d.md) "Deferred" + [`dd2d_spec.md`](dd2d_spec.md); roadmap
[`piginet_dd2d_plan.md`](piginet_dd2d_plan.md) steps 5–8):
- The **arrangement-complete negative certificate** (§8.4) — the schedule-critical hard item; until it
  lands, packing-infeasibility is provisional (`marginal(budget)`).
- Tier-1 off-the-shelf **PDDLStream + FastDownward** baselines.
- The full **7-variant two-tier** evaluation protocol beyond the shipped orderings.
- **Attack suites** — heuristic certificates H2–H4 + Tier-0 learned models.
- The **buffer-slack λ sweep** with bootstrap CIs (§11).
- The §9.5 filter-shift / §10.3 coverage audits and the held-out-generator split.
- The **PIGINet model / training / eval** steps.

> **Research-numbers gate:** because the negative certificate is deferred, **no label-dependent research
> numbers** are trustworthy yet — the EDA/heuristic outputs are diagnostic.

---

## 18. Quick-start commands

```shell
# always use the repo venv
.venv/bin/python -m blocks_tamp.dd2d.render_families                          # shape gallery -> out/dd2d/

# end-to-end demo (geometry-informed candidates planner)
.venv/bin/python -m blocks_tamp.dd2d.demo --lambda 0.6 --crowd 10 --num-problems 2 --max-videos 4

# the new A*/GBF geometric-distance heuristic path
.venv/bin/python -m blocks_tamp.dd2d.demo --planner pyperplan --pyperplan-search astar \
    --pyperplan-heuristic dist --num-problems 2 --max-videos 0

# geometry-blind difficulty EDA (stratified attempts-until-success + success prob)
.venv/bin/python -m blocks_tamp.dd2d.eda --episodes 200 --workers 8            # then eda_notebook.py

# 5-arm heuristic experiment (bfs / astar-hff / gbf-hff / astar-dist / gbf-dist)
.venv/bin/python -m blocks_tamp.dd2d.heuristic_experiment --smoke              # then heuristic_notebook.py

# collect a tiny PIGINet dataset (plumbing check), then inspect one record
.venv/bin/python -m blocks_tamp.dd2d.collect --smoke --out-root /tmp/dd2d_smoke
.venv/bin/python -m blocks_tamp.dd2d.inspect_example /tmp/dd2d_smoke/train

# tests
.venv/bin/python -m pytest blocks_tamp/tests/test_dd2d*.py -q
```
