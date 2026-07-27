# DD2D — implementation snapshot (2026-07-26)

A **standalone, current-state reference** for DD2D (Drawer Decluttering in 2D) as implemented in
`src/alphatamp/approaches/spectre/envs/dd2d/dd2d/`. It describes the objects, the generator, the grasp
model, the planners / refiner / heuristics, the packing certificate, the data-and-record layer, the
tooling, the tests, and what is still deferred — enough to understand or run the system without reading
the source.

> **Layout note.** DD2D lives at a **double-`dd2d`** path: `envs/dd2d/` is the SPECTRE-integration
> layer (the `spectre_*` adapters + domain), and `envs/dd2d/dd2d/` is the DD2D **core env** described
> here. All core modules run as `python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.<module>` (after
> `source .venv/bin/activate`, from the repo root).

This is a *snapshot*, not the design rationale. For the "why," see [`dd2d.md`](dd2d.md); for the full
spec, [`dd2d_spec.md`](dd2d_spec.md); for the CLI-argument tutorial, [`dd2d_demo_args_faq.md`](dd2d_demo_args_faq.md).
Units are **centimetres** throughout.

> **What changed since the 2026-07-08 snapshot.** DD2D was **migrated into the spectre package**
> (2026-07-12) — the old `blocks_tamp.dd2d.*` paths are gone, replaced by
> `alphatamp.approaches.spectre.envs.dd2d.dd2d.*`. The **grasp model was made physically realistic**
> (2026-07-24): fingers now slide on the two supporting lines' *true material contact runs* (no more
> closing onto air), and a new **internal-grasp** path reaches into concave regions (dumbbell bar,
> horseshoe opening, shoe corner); the curved `banana` became the blocky right-angled **`horseshoe`**.
> The **arrangement-complete negative packing certificate** (§8) — listed as *deferred* in the old
> snapshot — is now **built** (2026-07-18, `certificate.py`). The **collector** now guarantees **exact
> per-stratum counts**. New core modules: `certificate.py`, `demo_grasp_concave.py`, `harvest.py`. See
> `decisions.md` 2026-07-12 / 07-18 / 07-24. Anything in older notes referencing `blocks_tamp` paths or
> families like `bottle/board/mug/spray/spatula/lid` is out of date.

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

## 2. Module map (`envs/dd2d/dd2d/`)

| File | Role |
|---|---|
| `shapes.py` | Parametric shape library (7 families, 3 concave), `sample_shape`, isolation-graspability, holdout split. |
| `grasps.py` | Top-down parallel-jaw grasp model — **contact-run slides + internal concave grasps** (`grasp_cells`, `has_grasp`, `finger_rects`, `isolation_graspable`). |
| `world.py` | `DrawerScene` / `DrawerWorld` / `ItemState` / `StreamCounter`; the compaction buffer sampler `sample_buffer_pose`, `settle_pose`, `collar_pose`. |
| `scene.py` | Forward scene generator `generate_scene` (drawer/buffer geometry, target, collar, settled clutter). |
| `enumerate.py` | Geometric candidate enumeration (blocker sets → minimal sets → supersets → clearing + extraction re-checks). |
| `label.py` | Three-valued labeler (`feasible`/`infeasible`/`marginal`) + F1/F2/F3 filters; optional packing-certificate hook (`use_certificate`). |
| `certificate.py` | **Arrangement-complete negative packing certificate** (`certify_infeasible_by_packing`) — proves a subset cannot pack the buffer (§8). |
| `problem.py` | `DD2DProblem` + `generate_dd2d_problem` (generate → enumerate → label → filter → refiner-certify). |
| `refine.py` | `DD2DRefiner` — the shared backjumping refiner over `pick`/`place-buffer`/`retrieve`. |
| `planning.py` | `DD2DPlanner` (candidate enumerator) + `make_dd2d_planner` (candidates / pyperplan / symk, bfs/gbf/astar, heuristic wiring). |
| `heuristics.py` | Hand-written geometric **distance heuristic** (`distance_heuristic_factory`, forms `inv`/`avg`/`radius`). |
| `render.py` | matplotlib `render_scene` + `render_episode` (elevated-carry video) + `DD2DRenderBackend`. |
| `demo.py` | End-to-end demo CLI (generate → plan → refine → records + videos); stratum-pinned, parallel. |
| `demo_grasp_concave.py` | **Concave-grasp sanity demo** — the gripper closing on `dumbbell`/`shoe`/`horseshoe` internal features (videos). |
| `eda.py` | Geometry-blind pyperplan **difficulty EDA** (attempts-until-success + success prob). |
| `collect.py` | DD2D-native **dataset collector** — balanced strata, parallel, **exact per-stratum counts**. |
| `record_ext.py` | Geometry sidecar over the shared `record.PIGINetExample` (poses, shapes, boundary rings, crops). |
| `harvest.py` | Post-mortem **typed-fact harvest** from a failed `RefineResult` (the SPECTRE evidence pathway). |
| `inspect_example.py` | Single-record visualizer → a PIGINet-input figure. |
| `heuristic_experiment.py` | 5-arm (bfs / astar-hff / gbf-hff / astar-dist / gbf-dist) first-feasible-rank comparison. |
| `heuristic_notebook.py` | marimo dashboard for the heuristic experiment. |
| `eda_notebook.py` | marimo dashboard for the EDA. |
| `render_families.py` | Renders a labelled gallery PNG of the shape families. |
| `../domain/drawer_declutter.pddl` | Geometry-blind STRIPS domain (3 actions). |
| `__init__.py` | Public exports (matplotlib-free: `render` is intentionally *not* imported). |

`__init__.py` exports (unchanged, 19 names): `Shape, sample_shape, Grasp, grasp_cells, finger_rects,
grasp_cfree, has_grasp, isolation_graspable, ItemState, DrawerWorld, place_polygon, sample_buffer_pose,
DrawerScene, DD2DProblem, generate_dd2d_problem, make_dd2d_problem, DD2DRefiner, DD2DPlanner,
make_dd2d_planner`. Every module (including the heavier tooling — `collect`, `record_ext`,
`inspect_example`, `demo_grasp_concave`, the notebooks) runs from the repo root (venv active) as
`python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.<module>`.

**How DD2D feeds SPECTRE.** One level up, `envs/dd2d/` holds the SPECTRE-integration adapters:
`spectre_convert.py` (DD2D JSON records → SPECTRE `EpisodeRecord`s), `spectre_geometry.py` (rehydrate
exact grasp geometry from a stored `SceneGeometry`, *no regeneration*), `spectre_operators.py` (the
`relational_structs`/`bilevel_planning` substrate operator view), `spectre_render.py` (the shared
VLM-legible render), and `spectre_harvest.py` (offline post-mortem harvest). Migration provenance:
`MIGRATION_DD2D.md`. How SPECTRE consumes this data is documented in `docs/as_built_v2.2.md`.

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
| **`horseshoe`** | **concave** | spine `U(2.2,3.0)`, prong `U(2.5,3.0)`, arm `U(3.0,4.2)`, opening `U(2.8,3.8)` | **blocky C-opening**, symmetric, equal-length prongs (single 8-vertex polygon) |

Notes:
- **Concave set** = `{dumbbell, shoe, horseshoe}` (the `concave` flag). These are the "tricky convexity"
  shapes. They are built to **always be a single valid polygon**: `dumbbell`/`shoe` are Shapely
  `unary_union`s with a small `_OVERLAP = 0.4` cm so shared edges merge; `horseshoe` is constructed
  directly as one simple 8-vertex rectilinear polygon (spine + two equal prongs, opening on +x —
  **no annulus hole**, so validity is guaranteed). Prong thickness ≥ the finger width (2.5 cm) so a
  flat finger makes full-face contact (replaced the curved `banana`, whose grasps were only tangent
  points — see `decisions.md` 2026-07-24).
- **Sampling weights** (`_FAMILY_WEIGHTS`): `can` = 1.3, `box` = 1.3, all others = 1.0.
- **Isolation-graspability:** every sampled shape must admit ≥1 grasp *in isolation* (some direction with
  width ≤ 12 cm aperture and a non-empty contact-overlap interval); non-graspable draws are resampled
  (`sample_shape(..., require_graspable=True)`, up to `max_tries=40`).
- **`Shape` dataclass** (frozen): `family`, `polygon` (centroid at origin), `concave`; properties `size`
  (bbox w,h), `area`, `r_max` (max centroid-to-boundary distance).
- **`split` / holdout:** `sample_shape(rng, family=None, split="train")`. `split="holdout"` scales every
  dimension band by **×1.15** and swaps `bowl → can`. (Wired but not yet used in a formal train/test
  experiment.)
- Preview the families: `python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.render_families` →
  `out/dd2d/shape_families.png`.

---

## 4. Grasp model (`grasps.py`)

A **top-down parallel-jaw grasp** = two finger rectangles pressed flush against opposite sides of an
item, both collision-free. The model was **rebuilt 2026-07-24** so the fingers land on *actual
material* (no closing onto air) and can reach into concave regions (`decisions.md` 2026-07-24).

**Constants:** `FINGER_WIDTH = 2.5` (tangential, along the supporting line), `FINGER_THICK = 2.0`
(normal), aperture `[MIN_APERTURE 0.5, MAX_APERTURE 12.0]`, `N_DIRECTIONS = 18` (10° steps over
`[0, 180)`), `N_SLIDES = 5`, interior fraction `_INTERIOR = 0.80`; and for the internal path
`_FULL_FACE_FRAC = 0.9`, `_SCAN_STEP = 0.4` cm.

A **`Grasp`** (frozen) is `(alpha, s, xmin, xmax)`: `alpha` = grasp direction (item-frame radians),
`s` = the finger slide (a `y` in the −`alpha` rotated frame), `xmin`/`xmax` = the two supporting lines
(finger x-positions); `width = xmax − xmin` is the aperture.

**Contact-run construction (both fingers on material, gap 0).** In the −`alpha` frame the fingers sit
at the item's x-extremes `xmin`/`xmax`. `direction_admissible(shape, alpha)` returns
`(ok, xmin, xmax, slide_intervals)`, where `slide_intervals` is the **intersection of the two
supporting lines' *actual* contact runs** — `_contact_runs_on_line` returns every disconnected `y`-run
where a vertical line meets the footprint, and `_intersect_runs` keeps the `y` where *both* fingers
touch material. This replaced the old `y`-*hull*, which let a finger straddle a gap (a C-opening /
waist) and close onto air. A single tangent point counts (gap 0), so a circle keeps its one valid
grasp; `_slide_positions` then distributes up to `N_SLIDES` slides across the real runs (interior 80%).

**Internal concave grasps.** Beyond the outer envelope, `_internal_grasps` runs a **scan-line
antipodal** enumerator (`_SCAN_STEP` horizontal lines) that admits any strictly-*internal* flat
material segment as a grasp iff **(a) finger-fit** — the finger rects clear the item's own material
(the grippers fit into the concavity) — and **(b) full-face contact** — each finger inner face lies on
the boundary for ≥ `_FULL_FACE_FRAC · FINGER_WIDTH` (`_face_contact_len`). Requiring full-face flat
contact excludes curved-shape *sliver* pinches (a circle gets no internal grasp). `grasp_cells(shape)`
= the global-envelope grasps **+** internal grasps (the internal path skips the global segment, so
they never duplicate). Families that gain internal grasps: **dumbbell** (the bar/waist), **horseshoe**
(spine + into the C-opening between prongs), **shoe** (the L-corner arm); the convex/curved families
(`can`/`bowl`/`box`/`pillcase`) gain **none**.

**API:** `grasp_cells(shape)` (all `(α, slide)` cells, no collision filtering); `finger_rects(g, pose)`
(world-frame finger rectangles); `grasp_cfree(g, pose, obstacles)` (boundary-touch allowed, penetration
not); `has_grasp(shape, pose, obstacles)` (first collision-free cell, or `None`);
`isolation_graspable(shape)` (the resample gate — true iff some direction is admissible), used by the
shape sampler; `direction_admissible(shape, alpha)` (the 4-tuple above).

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

## 8. Labeler (`label.py`) + the packing certificate (`certificate.py`)

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

### 8.1 The arrangement-complete negative certificate — **now built** (2026-07-18)

The old snapshot listed this as *deferred*; it is now implemented in `certificate.py` (spec §8.4).
`certify_infeasible_by_packing(scene, subset)` is **three-valued**: `True` = **provably**
infeasible-by-packing, `False` = a packing was found (⇒ not infeasible), `None` = undecided within
budget (⇒ stays marginal). It is **sound — zero false-infeasible by design** (any doubt weakens to
`None`, never to `True`). Pipeline (cheap → expensive): (1) δ/2-deflated shapes; (2) the H1 area bound
on **exact** deflated areas; (3) for `|S| ≤ MAX_ORDER_ITEMS = 5`, an exact **arrangement DFS** over
**all** placement orders — exact convex decomposition (Shapely `constrained_delaunay_triangles`, which
makes the concave families sound), exact NFP/IFP, and a Lipschitz rotation grid `Δθ = δ/(4·r_max)`.
Budgets `DEFAULT_EGE_BUDGET = 100_000`, `DEFAULT_TIME_BUDGET_S = 5.0` (timeout ⇒ `None`). Verified 0
false-infeasible over ~730 constructed-feasible packings.

**How it plugs in:** `label.py`'s `label_candidate` / `label_all` take **`use_certificate` (default
off)**. When on, a provisional `marginal(budget)` is upgraded to **`infeasible(packing)`** if the
certificate proves it, relabeled **`marginal(inaccessible)`** if it instead finds a packing, or left
`marginal(budget)` on `None`. It is **off in the default generation and collection paths** (both label
feasibility via the refiner's real outcomes — see §9), because the generation rejection-loop only needs
the feasible labels; the certificate is meant for authoritative once-per-candidate (re)labeling. The
label-side H1 area bound and the `not extractable ⇒ infeasible(extraction)` rule are always on,
independent of the flag.

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

## 10. Planners (`planning.py` + generic `envs/dd2d/planning.py`)

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

All run from the repo root with the venv active, as
`python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.<module>`. Key flags + defaults:

### `demo.py` — end-to-end demo (records + videos)
`--num-items` (None→9–14), `--lambda` (0.8), `--margin` (1.0), `--split {train,holdout}` (train), `--k`
(12), `--seed` (0), `--num-problems` (1), `--order {published,random,slack,oracle}` (published),
`--planner {candidates,symk,pyperplan}` (candidates), `--pyperplan-slack` (none),
**`--pyperplan-search {bfs,gbf,astar}` (bfs)**, **`--pyperplan-heuristic {hff,hadd,dist,dist-avg,dist-radius}` (hff)**,
`--crowd` (10), `--diverse-crowd` (flag), `--require-subset` (flag), `--min-subset` (None; **implies
`--require-subset`**; require-subset alone = floor 2), `--max-stream-calls` (300), `--retry-cap` (10),
`--samples-per-step` (15), `--time-budget` (None), `--no-certify` (flag), `--max-videos` (6),
`--video-format {mp4,gif}` (mp4), `--out-dir` (out_dd2d).

**New flags (2026-07-24):** `--stratum {0,1,2,3}` (pin the exact min-feasible-subset size; overrides
`--require-subset`/`--min-subset` and resamples until it finds problems of exactly that stratum),
`--min-blockers` / `--max-blockers` (sample the blocker count from a range instead of `--num-items`),
and `--workers` (default 1; when > 1, **runs the problems in parallel** via `ProcessPoolExecutor` —
worker-count-invariant, disjoint seed slots so serial and parallel give the same set).

### `demo_grasp_concave.py` — concave-grasp sanity demo (NEW)
Renders short videos of the gripper closing on the concave families (`dumbbell` bar, `horseshoe`
C-opening, `shoe` corner), tagging the internal-region grasps — the visual proof that the 2026-07-24
grasp model contacts material (no floating fingers) and reaches into concavities. Writes to
`out_dd2d/grasp_demos/`.

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

### `collect.py` — DD2D-native dataset collector
Balanced min-subset **strata (0, 1, 2, 3)**, parallel, planned with **astar + dist**. Per problem:
generate a requested-stratum instance → plan k skeletons → refine and persist. **Exact per-stratum
counts (2026-07-24):** an **in-flight cap** (`kept + in_flight ≤ target` per stratum) prevents overshoot
and diverts freed workers to under-target strata, and a `_truncate_to_targets` finalization guarantees
**exactly** the sub-target — so `--target-train 400` yields **exactly 100/100/100/100** (`decisions.md`
2026-07-24). **`full_pool=True`** (default) refines *all k* plans (many positive + negative per problem,
no length confound); legacy mode stops at first feasible and drops unsolved problems. `DD2DCollectConfig`:
`lam=0.8, margin=1.0, crowd=5, diverse_crowd=True, k=200, budget=None` (time-governed), `retry_cap=10,
samples_per_step=15, time_budget=20.0, full_pool=True` (stratum 0 forces `crowd=0`). CLI: `--out-root`
(data/dd2d/raw), `--workers` (8), `--target-train` (400), `--target-test` (100), `--target-val` (100),
`--splits` (train,test,val), `--band` (1_000_000), `--crowd` (5), `--lambda` (0.8), `--time-budget`
(20.0), `--resume`, `--smoke` (→ target 3 = 1/stratum, ≤ 2 workers). `k` and `full_pool` are config-only
(no CLI override). Output: `<split>/<problem_id>/{images/, scene.png, NNN.json}` + per-split
`manifest.json` + `attempted.log` (the resume log); a `BrokenProcessPool` (OOM) is survived by
finalizing the split and prompting `--resume`.

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

Under `envs/dd2d/tests/` (collected counts; parametrized tests expand beyond one `def`):

| File | Count | Coverage |
|---|---|---|
| `test_dd2d.py` | 55 | shapes/grasps (incl. contact-run + internal-grasp invariants), buffer sampler + world, scene/generation, crowd/collar/subset, candidates/labels, planners (incl. the gbf/astar heuristic arms), refiner, record/render. |
| `test_dd2d_collect.py` | 14 | full-pool vs legacy collection, drop-unsolvable, **exact-per-stratum truncation** + resume, determinism, band disjointness, manifest/disk layout. |
| `test_certificate.py` | 16 | **(NEW)** the negative packing certificate — soundness (zero false-infeasible, incl. tight/near-threshold batteries), convex-cover exactness, verdict mapping. |
| `test_demo_grasp_concave.py` | 12 | **(NEW)** every grasp cell makes material contact, full-face horseshoe grasp, internal grasps on dumbbell/horseshoe/shoe, convex families have none. |
| `test_dd2d_record_ext.py` | 2 | crop-per-object PNGs, geometry-augmented example round-trip. |
| `test_dd2d_inspect.py` | 2 | record → figure, record resolution from a split dir. |

(`test_harvest.py` (5) is adjacent — it tests the post-mortem harvest, not the core env geometry.)

Run all DD2D env tests from the repo root (venv active):
`python -m pytest src/alphatamp/approaches/spectre/envs/dd2d/tests/` (or select
`tests/test_dd2d*.py tests/test_certificate.py tests/test_demo_grasp_concave.py`).

---

## 17. Status — implemented vs. deferred (2026-07-26)

**Implemented:** the world/grasp/label/record/render layer; the **rebuilt grasp model** (contact-run
slides + internal concave grasps); the forward generator with crowd/diverse-crowd/unblocked-target
strata and the F1–F4 filters + refiner-certification; candidate enumeration; the three-valued labeler;
the **arrangement-complete negative packing certificate** (`certificate.py`, sound + tested — this was
the schedule-critical deferred item); the shared backjumping refiner; **all three planner families**
including the geometric **A\*/GBF distance-heuristic** path; the geometry-blind difficulty **EDA**; the
5-arm **heuristic experiment**; the DD2D-native **dataset collector** (now exact-per-stratum); the
geometry **record sidecar**, record **inspector**, **concave-grasp demo**, and post-mortem **harvest**.

**Deferred** (see [`dd2d.md`](dd2d.md) "Deferred" + [`dd2d_spec.md`](dd2d_spec.md)):
- **Enabling the certificate in generation/collection** — it is built and callable but `use_certificate`
  is **off by default** (§8.1); the shipped collector labels feasibility from the refiner's real
  outcomes. Turning it on for authoritative once-per-candidate relabeling is the remaining integration.
- Tier-1 off-the-shelf **PDDLStream + FastDownward** baselines.
- The full **7-variant two-tier** evaluation protocol beyond the shipped orderings.
- **Attack suites** — heuristic certificates H2–H4 + Tier-0 learned models.
- The **buffer-slack λ sweep** with bootstrap CIs (§11).
- The §9.5 filter-shift / §10.3 coverage audits and the held-out-generator split.

> **Label caveat.** The negative-certificate blocker is resolved, but two things bound label-dependent
> DD2D numbers: (1) the default collector uses **refiner-outcome** labels (a plan is feasible iff the
> refiner actually solved it — sound, but not the certificate's *proven*-infeasible), and (2) the
> **2026-07-24 grasp fix shifted feasibility labels**, so any pre-fix collection is stale. The current
> authoritative grasp-fixed collection is what SPECTRE trains on — see `docs/as_built_v2.2.md` for the
> downstream results.

---

## 18. Quick-start commands

```shell
# from the repo root, activate the venv first
source .venv/bin/activate
M=alphatamp.approaches.spectre.envs.dd2d.dd2d          # module prefix

# shape gallery -> out/dd2d/
python -m $M.render_families

# concave-grasp sanity videos -> out_dd2d/grasp_demos/
python -m $M.demo_grasp_concave

# end-to-end demo (geometry-informed candidates planner)
python -m $M.demo --lambda 0.6 --crowd 10 --num-problems 2 --max-videos 4

# stratum-pinned, parallel demo
python -m $M.demo --stratum 3 --num-problems 4 --workers 4 --max-videos 0

# the A*/GBF geometric-distance heuristic path
python -m $M.demo --planner pyperplan --pyperplan-search astar \
    --pyperplan-heuristic dist --num-problems 2 --max-videos 0

# geometry-blind difficulty EDA (stratified attempts-until-success + success prob)
python -m $M.eda --episodes 200 --workers 8            # then eda_notebook.py

# 5-arm heuristic experiment (bfs / astar-hff / gbf-hff / astar-dist / gbf-dist)
python -m $M.heuristic_experiment --smoke              # then heuristic_notebook.py

# collect a tiny dataset (plumbing check), then inspect one record
python -m $M.collect --smoke --out-root /tmp/dd2d_smoke
python -m $M.inspect_example /tmp/dd2d_smoke/train

# tests
python -m pytest src/alphatamp/approaches/spectre/envs/dd2d/tests/ -q
```
