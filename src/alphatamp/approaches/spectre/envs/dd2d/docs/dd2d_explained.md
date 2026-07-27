# DD2D, demystified — a standalone explainer

*A self-contained overview of the DD2D environment as implemented, written for a reader who
has not seen the code. It explains what the task is and why it is designed this way, then
answers three specific questions: how objects are generated, how the refiner works, and how the
generator places targets/blockers, guarantees the "needs a subset" property, and guarantees
everything fits in the drawer.*

All code lives in `blocks_tamp/dd2d/`. Units are **centimetres** throughout.

---

## 1. What DD2D is, in one paragraph

DD2D ("Drawer Decluttering in 2D") is a top-down 2D manipulation task. A rectangular **drawer**
holds 9–14 rigid household items; one is the **target**, which starts **ungraspable** because its
neighbours block every two-finger grasp. Beside the drawer sits a small **buffer** (an open
counter strip). The robot must **stage a subset of the blocking items onto the buffer** to open a
grasp on the target, then retrieve it. The only interesting decision is *which subset to move*:
almost every symbolically-valid plan is geometrically infeasible (the chosen items don't fit the
buffer, or removing them still doesn't clear the target), while the correct ones work easily.

**Why this design.** The research question is whether plan-feasibility is predictable from cheap,
low-order features (which is what learned TAMP accelerators like PIGINet assume). DD2D is built so
that feasibility hinges on a **global, continuous, high-interaction-order geometric statistic** —
whether a *chosen subset* jointly packs into a limited buffer *and* is grasp-accessible — which is
exactly the kind of signal a low-order classifier should struggle with. The whole environment is
engineered so that this difficulty is **measured on a naturalistic distribution, not hand-installed
into specific instances** (see §6).

The world has three regions (all Shapely polygons): the **drawer** interior, a 1.5 cm **wall band**
ringing it (a first-class obstacle for grasp fingers), and the **buffer** strip 6 cm to its right.
The buffer has no walls — only already-staged items obstruct it.

---

## 2. How objects are generated  *(`shapes.py`)*

Items are **not** hand-drawn puzzle pieces; each is sampled from one of **7 parametric families**
anchored to real product sizes, then polygonised to a Shapely polygon in its own frame (centroid
at the origin, so a pose `(x, y, θ)` places it by rotate-about-centroid then translate):

| Family | Shape | How it's built |
|---|---|---|
| can | circle (small, 4–8 cm) | regular polygon, 28 vertices |
| bowl | circle (large, 8–12 cm) | regular polygon, 28 vertices |
| box | rectangle | 50% sharp corners, 50% rounded; small..large |
| pillcase | capsule | rectangle + two semicircular end-caps |
| **dumbbell** | two blocks **∪ bar** | end blocks joined by a thinner/longer bar → **concave waist** |
| **shoe** | two rects in an **L** | similar-sized rectangles at a right angle → **concave corner** |
| **horseshoe** | blocky **C** | a spine + two equal-length prongs with an opening → **concave C** |

So the concave shapes (dumbbell/shoe/horseshoe) are produced by **taking the union of simple
primitives** (or, for the horseshoe, one 8-vertex rectilinear C polygon), which yields genuinely
non-convex footprints. Those three families carry a `concave=True` flag; every downstream result can
be reported stratified on it. Family choice is weighted (boxes/cans slightly upweighted); dimensions
and a little shape-noise are sampled per item.

Two guarantees are enforced at sampling time:
- **Graspable in isolation.** Every sampled shape must admit at least one valid two-finger grasp on
  its own (some direction whose width ≤ the 12 cm gripper aperture with a real finger contact);
  otherwise it is **resampled**. This is why, e.g., a `bowl` whose diameter happens to exceed
  12 cm is thrown back, and why the concave families (which lose most grasp directions) still keep ≥ 1.
- **Fits the drawer sensibly.** A `box` larger than 45 % of the drawer's short-side² is
  rejected (it would dominate the scene).

A held-out split (dimension ranges shifted ±15 % and one family swapped) exists for generalisation
tests; it is wired but not exercised in the default pipeline.

---

## 3. How the environment is generated  *(`scene.py`, `enumerate.py`, `label.py`, `problem.py`)*

Generation is **forward-generate-then-label**: build a naturalistic scene, then *measure* its
properties with an exact geometric analysis, and keep it only if it passes a few filters. This is
the crux of the environment, so it is worth being precise.

### 3.1 Placing the target and blockers — and why everything fits

`generate_scene` samples the drawer (`W∼U[35,50]`, `D∼U[28,40]`), the buffer (scaled by the
difficulty dial `λ`), a target **fill fraction** `f∼U[0.35,0.55]`, and an item count `N∼U{9..14}`.

- **Target** — one item, placed with a random rotation at a uniform position in the **central
  50 %×50 %** of the drawer, accepted only if its footprint is fully inside the drawer.
- **Blockers** — added by a **settled-clutter** procedure: sample an item, drop it at a uniform pose,
  slide it toward the nearest contact along a random direction, back off slightly. It is added
  **only if** its footprint is both **contained in the drawer** and **non-overlapping** with every
  item already placed; after 30 failed tries the item is skipped. Items keep being added until the
  fill fraction `f` is reached or `N` is hit.

**How we guarantee all items fit in the drawer:** this is pure **rejection sampling on the
geometry**. No placement is ever accepted unless it is *inside the drawer and collision-free*, so by
construction the final scene is a set of non-overlapping footprints inside the drawer — overlaps are
impossible, not merely unlikely. The fill cap (≤55 % of drawer area) leaves headroom so the procedure
reliably finds room. Nothing is "made to fit" after the fact; candidates that don't fit are simply
never accepted.

### 3.2 The collar (crowding) prior — how "needs a subset" is created

Left alone, the settled clutter almost always leaves the target with an *open* grasp direction whose
one finger lies in a gap; removing the single item blocking the *other* finger clears the target — so
**one object suffices** ~90–95 % of the time (measured). To make problems that genuinely require a **2+ item
subset**, the target must be **pincered**: every admissible grasp direction straddled by two distinct
items, so no single removal opens any corridor.

The `--crowd N` knob does exactly this (`collar_pose` in `world.py`): after the target is placed
(and biased to a compact round shape so it can be fully ringed), it places `N` "collar" items that
slide **inward toward the target** from evenly-spaced, opposing bearings — the opposite of the default
random-bearing settle. Opposing collar items bracket each finger corridor, so the smallest set that
clears the target becomes a **diametrically-opposite pair** (or larger). `crowd=10` yields ~50 % of
problems needing a subset; `crowd=0` is the naturalistic ~5 % baseline.

### 3.3 How the min-feasible-subset requirement is *guaranteed* (measured, then filtered)

We never hand-build the required pair. Instead we **enumerate all clearing subsets, label each one
for feasibility, and read off the minimum feasible size** — then optionally filter on it:

1. **Enumerate** (`enumerate.py`). For each of the target's 18×5 grasp cells, compute the set of
   items whose footprints hit that grasp's two finger rectangles = a **blocker set**. Take the
   **minimal blocker sets under set-inclusion** (these are the minimal clearing subsets, of *any*
   size — a pincered target naturally yields size-2 minimal sets), grow them with a few adjacent
   items, and keep two exact re-checks per candidate: (a) with the subset removed the target actually
   has a clear grasp, and (b) the subset can be removed from the drawer in *some* order (each member
   graspable at its turn).
2. **Label** (`label.py`). Each candidate is labelled **feasible** iff an extraction order exists
   **and** a real geometric packing search finds an *accessible* δ-clearance packing of the subset
   into the buffer (packs with margin, and each item is graspable as it is placed); **infeasible** if
   it can't be extracted or a sound area bound proves it can't fit; else **marginal**.
3. **Measure.** `min_feasible_subset` = the size of the **smallest feasible clearing subset**. If it
   is 1, some single object both clears the target and packs → one object suffices. If it is ≥2, every
   single-object removal fails (target stays blocked, or that item is itself un-graspable/doesn't
   pack) and only a subset works — *this is the property, computed exactly, not asserted.*
4. **Filter/guarantee.** The `--require-subset` flag turns this into a hard guarantee: the generator
   **resamples until** `min_feasible_subset ≥ 2` (rejection sampling on the measured property). Off by
   default, so the distribution is a natural mix whose subset-fraction is reported.

So "guaranteeing the min-feasible-subset requirement" = **the collar makes it common, the exact
enumerator+labeler measures it, and the optional filter rejects anything that doesn't meet it.** The
difficulty is a real geometric property of each kept scene, verified by the same packing machinery the
refiner uses — never installed by construction.

### 3.4 The acceptance filters and certification

Every candidate scene must pass three **decision-relevance filters** (evaluated after labelling; the
scene is resampled if any fails):

- **F1 — target blocked:** the target has no clear grasp in the initial scene (otherwise the task is
  trivial).
- **F2 — a real choice:** ≥2 distinct minimal clearing subsets exist (so the decision is non-trivial).
- **F3 — solvable:** ≥1 candidate is confidently feasible (there *is* a solution).
- **F4 (optional):** `min_feasible_subset ≥ 2` (the subset guarantee of §3.3).

Finally, **certification** runs the *real refiner* on the kept scene: the intended plan (staging the
smallest feasible subset) must refine successfully, and the degenerate "just grab the target" plan
must fail. This guarantees each kept instance is genuinely solvable *under the refinement budget you
collect with*, and that the target is truly blocked.

---

## 4. How the refiner works  *(`refine.py`, `world.py`, `grasps.py`)*

The refiner takes a **staging skeleton** — `pick(o₁); place-buffer(o₁); … ; retrieve(target)` — and
tries to bind continuous poses/grasps to it, replaying it against a mutable world. Two observations
you made ("good at fitting things in comfortably" and "good at picking things up") come directly from
two deliberately non-trivial samplers.

### 4.1 Grasping — why it reliably picks items up  *(`grasps.py`)*

The gripper is a top-down parallel jaw. A candidate grasp is a pair `(direction, slide)` from an
**18-direction × 5-slide grid** (18 approach angles over [0°,180°), 5 positions along the contact
edge). A direction is *admissible* only if the item's width across it is ≤ the 12 cm aperture and the
two jaw sides actually make overlapping contact. Each grasp materialises as **two finger rectangles**
(2.5 cm × 2.0 cm) flush against opposite sides of the item. A grasp is **valid at a given pose** iff
both finger rectangles are clear of all *other* items and the wall band.

So `pick` isn't a fixed top-down clamp — the refiner **searches up to 90 grasp cells and returns the
first whose fingers are collision-free**. That is why it "finds a way to pick things up": if any
approach angle threads between the neighbours, it finds it; only a genuinely buried item (no clear
angle) fails. The same search is what decides whether the target is blocked (F1) and whether a staged
item is accessible in the buffer.

### 4.2 Placing on the buffer — why it packs comfortably (it is *not* uniform)  *(`sample_buffer_pose`)*

This is the key to your "things fit within the tray comfortably" observation. The buffer-pose sampler
is **compaction-biased**, not uniform. For each placement it:

1. draws `m_p = 15` candidate poses, most of them by **contact proposals** — start the item abutting
   an already-staged item or the buffer edge (not at a random point);
2. **pushes each candidate bottom-left** — greedily slide it in −x then −y until it just contacts
   something, so it nestles into the corner/against its neighbours;
3. scores every collision-free, in-buffer candidate by `x + 0.01·y` (plus a little Gumbel tie-noise)
   and returns the **lowest-scoring** one — i.e. the most bottom-left, most tightly packed pose.

Because it always compacts toward a corner and packs against existing items, it fills a tight buffer
far better than uniform sampling would — feasible subsets reliably pack, and *in*feasible ones
reliably fail to (which is what makes the feasibility labels faithful). Crucially the sampler is still
deliberately **cheap and incomplete**: on a subset that truly can't fit it can only *fail to find*,
burning budget — that expensive failure is the cost structure under study, not a bug.

`m_p` (candidate poses per call, "sampler strength") is a tunable knob; raising it improves packing
quality without spending more "stream calls."

### 4.3 The search: accessibility + backjumping + budgets

For each `place-buffer` step the refiner samples a buffer pose (above) **and** re-checks
**accessibility** — that the item is still graspable at that pose clearing the already-staged items
(fingers may overhang the wall-less buffer edge). It retries up to `retry_cap` (default 10) times; if
the step stays stuck it **backjumps** — undoes the previous placement and re-samples it — so a bad
early pose can be revised. `retrieve(target)` at the end must find a clear target grasp.

The whole refinement is bounded by a **budget**, tunable three ways (all exposed on the demo):
- `--budget` — total *stream calls* (sampler + test invocations), the global effort cap;
- `--samples-per-step` — `m_p`, the sampler-strength dial of §4.2;
- `--retry-cap` — attempts per step before backjumping;
- `--time-budget` — a wall-clock cap per plan (you can run purely by time + per-step by disabling the
  call cap).

Whether the whole skeleton binds within the budget is the **(noisy) feasibility label** written to each
training example — exactly the label a PIGINet-style classifier would learn to predict.

---

## 5. The planners  *(`planning.py`, `dd2d/planning.py`)*

The environment separates *proposing* candidate plans from *refining* them. The symbolic domain is
deliberately **geometry-blind** (`pick`, `place-buffer`, `retrieve` only; no notion of "blocks" or
"fits"), so the shortest plan is literally "grab the target," and longer plans stage arbitrary items.
Two planner tiers use this:

- **`candidates` (default, geometry-informed).** Enumerates the clearing subsets from §3.3 (which it
  knows from the grasp geometry) and turns each into a staging skeleton, ordered shortest-first. It
  puts the feasible plan near the top of the list (typically rank 1–5).
- **`pyperplan` / `symk` (standard, geometry-blind baselines).** Off-the-shelf diverse planners that
  enumerate the `k` shortest symbolic plans ascending in length. Being blind to which subset actually
  clears the target, they must enumerate and refine *many* wrong plans before reaching a feasible one
  (e.g. rank ~85/200 for a 2-item subset; rank ~691/800 for a 3-item subset). `pyperplan` and `symk`
  are two engines for the same "top-k diverse plans" goal (SymK is a faster symbolic top-k planner;
  `pyperplan` is a dependency-light BFS fallback).

The **gap** between "candidates ranks the solution ~2nd" and "a standard planner refines dozens–
hundreds of plans first" is the intended research finding: on this distribution, *ranking* plans (what
PIGINet does) pays off, precisely because feasibility is a global packing property the symbolic layer
can't see.

---

## 6. Why it works — the design in one place

Everything above serves one principle: **the difficulty is measured on a naturalistic distribution,
not installed into rigged instances.**

- **Objects** are sampled from realistic parametric families (incl. genuinely concave dumbbells/shoes/
  horseshoes), with only two honest filters (graspable-in-isolation, not-too-huge).
- **Scenes** are built by rejection sampling that *cannot* produce overlaps or out-of-drawer items, so
  "everything fits" is guaranteed by construction, and the crowding prior makes multi-item clearing
  *common* without dictating which items.
- **The hard property** (needs a 2+ subset) is not asserted — it is **computed exactly** by enumerating
  clearing subsets and labelling each with the *same* real packing + grasp machinery the refiner uses,
  then optionally filtered on. So a "requires a subset" label is a true geometric fact about the scene.
- **The refiner succeeds** on feasible plans and faithfully fails on infeasible ones because its two
  samplers are good but honest: a grasp search that finds any clear approach (good at picking up), and
  a compaction packer that fills tight buffers (good at fitting) — while staying cheap enough that a
  genuinely infeasible subset still costs real budget to reject.

That combination — realistic objects, containment-by-construction, measured-then-filtered difficulty,
and an honest geometry-informed refiner — is why DD2D produces clean, non-trivial, correctly-labelled
instances.

---

## Appendix — key parameters and where the code lives

| Concept | Value | File |
|---|---|---|
| Drawer W × D | `U[35,50] × U[28,40]` cm | `scene.py` |
| Buffer L × d (× λ) | `U[25,45] × U[12,20]` cm, 6 cm right | `scene.py` |
| Wall band | 1.5 cm ring | `scene.py` |
| Fill fraction `f`, item count `N` | `U[0.35,0.55]`, `U{9..14}` | `scene.py` |
| Shape families | 7 (3 concave) | `shapes.py` |
| Grasp grid | 18 directions × 5 slides; aperture ≤ 12 cm; fingers 2.5 × 2.0 cm | `grasps.py` |
| Buffer sampler | `m_p = 15` compaction candidates, bottom-left push | `world.py` |
| Candidate cap / adjacency | ≤ 40 candidates; supersets within 2 cm | `enumerate.py` |
| Label margin δ | 1.0 cm; packing restarts = 3 | `label.py` |
| Refiner budget defaults | B = 300 stream calls, retry-cap = 10, m_p = 15, time = ∞ | `refine.py` |
| Crowd (subset difficulty) | `--crowd 10` ⇒ ~50 % require a subset; `0` = naturalistic | `scene.py` / `problem.py` |
| Buffer scale λ | difficulty dial (tighter buffer = harder packing) | `problem.py` |

Runnable example:

```shell
.venv/bin/python -m blocks_tamp.dd2d.demo --lambda 0.6 --seed 0 --num-problems 10 --crowd 10
```

For the full module-by-module design and the deferred research pieces, see `docs/dd2d.md`; the
original specification is `docs/dd2d_spec.md`.
