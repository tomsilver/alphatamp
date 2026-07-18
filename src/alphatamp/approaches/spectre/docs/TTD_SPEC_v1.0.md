# Tote-and-Tray Decluttering (TTD): Implementation Specification

**Version:** 1.0 (post-revision: stacking excluded, compaction-biased placement stream, three-dial pilot)
**Target simulator:** PyBullet (Drake port optional)
**Audience:** an implementer with TAMP/PDDLStream familiarity but no access to prior design discussions. This document is self-contained: it explains *why* every design element exists, then specifies *how* to build the environment, the planted instance generator, the planner/refiner stack, and the video-evidence pipeline.

---

## 1. Purpose and research context

### 1.1 The research gap this environment targets

Task-and-motion planning (TAMP) systems separate planning into (a) a discrete **skeleton** (a sequence of symbolic actions with continuous parameters left free) and (b) **refinement** (sampling/optimizing the continuous parameters — grasps, placements, motions — via "streams"). Recent learning-for-TAMP systems accelerate this loop:

- **PIGINet** (Yang et al., 2023) learns a feasibility classifier `f(image, plan, goal) → [0,1]` over *complete* skeletons and reranks the candidates that a diverse symbolic enumerator produces. It cannot generate skeletons itself, and its inputs are the initial scene image (per-object CLIP crops), goal literals, and the plan token sequence.
- **LAZY** (Khodeir et al., 2023) learns a goal-conditioned graph-attention policy over a scene graph (objects + predicates + fixed geometric features) used as a per-step prior inside Levin Tree Search, with online success/attempt statistics per node and computation-graph-key transfer of refinement work across skeletons that share structure.

TTD is designed so that both mechanisms are stressed by construction: skeleton feasibility depends on a **global, high-interaction-order, continuous geometric statistic** (whether a *subset* of irregular shapes jointly nests into a tight region) that (i) is not expressible as any small fixed-arity predicate set, (ii) is not decomposable into per-object or pairwise terms, (iii) has no cheap heuristic certificate, and (iv) reveals infeasibility only *late* in refinement, so every wrong skeleton is expensive and online statistics learn slowly. At the same time, every instance is **solvable by construction** (a planted witness with margin), refinement of correct skeletons is easy, and an oracle that knows the planted answer solves each instance in roughly one refinement attempt — so the benchmark measures *skeleton-selection efficiency*, not solvability.

### 1.2 Design properties the environment must satisfy

These are stated here as requirements; every subsequent section references them by ID.

- **C1 — High-order coupling.** The discriminating statistic for skeleton feasibility is `packs(S, tray)`: joint 2D nesting of a k-object subset (k ∈ {4,5}) of concave footprints, with continuous rotations, into a tray sized only slightly larger than the subset. This is an order-k continuous interaction. Exact certification is NP-hard 2D irregular nesting; nothing pairwise or aggregate (areas, counts, bounding boxes) suffices. Corollary constraints: the *symbolic* layer must never encode packing or blocking geometry (Sections 5–6), and shapes must be concave (Section 4.2) because convex-shape packing is largely predicted by low-order summaries and solved by greedy heuristics.
- **C2 — Concentrated, bimodal difficulty.** Every enumerated candidate skeleton either refines with probability ≥ 0.9 or ≤ 0.1 (over stream random seeds, within a fixed stream-call budget B). Difficulty lives entirely in *choosing* the subset, not in executing it. Achieved via planted margins (Section 8) plus a checker-level label rule (Section 7.3) plus regeneration of marginal instances.
- **C3 — No degenerate shortcut.** No universal subset works across instances; no conservative "stage everything" plan exists (tray is strictly smaller than the tote contents); no in-tote shuffling shortcut exists (Section 8.5); stacking on the tray is excluded by the manipulation model (Section 4.3.4). Wrong greedy choices consume tray space irreversibly within an execution episode, while *search-time* backtracking remains unrestricted.
- **C4 — Learnable structure.** Instances are generated with regularities a sufficiently expressive model (e.g., a set transformer over subset shape encodings) can learn; a bracketing pilot verifies low-order models sit near chance while the planted oracle sits near 1.
- **C5 — Costly rejection, no cheap certificates.** Wrong subsets fail only at the *last* tray placement after k−1 successes; cheap hand-coded certificates (summed area, one-shot greedy insertion, bounding-box bounds) must be unreliable (verified — Section 10.3). A compute-accounted "checker-in-loop" baseline that calls an anytime exact nester at plan time is a required comparison bracket, not a forbidden move.
- **C6 — Physical realizability within a verified window.** Everything runs in PyBullet under a quasi-static SE(2)×lift manipulation model. The critical empirical risk is that C2's ceiling (a shared sampling refiner must succeed on planted subsets ≥ 90% of the time) and C5's floor (cheap heuristics must fail) may not coexist; Section 10 defines the three-dial pilot (tray slack × plant margin δ × sampler strength) whose pass/fail conditions decide this before full dataset construction.

Verification gates (G1: an off-the-shelf backtracking planner solves ≥ 95% of instances at generous budget; G2: at practical budget the same planner burns 5–10 failed refinements per solve vs. ~1 for the oracle; G3: anti-leakage audits on the planted generator; G4: fair incumbent baselines) and reported diagnostics (D1: enumeration rank of first feasible candidate; D2: divergence-to-failure depth and cost per rejection; D3: retrieval-baseline performance; D4: generalization axes; D5: per-heuristic residual error) are operationalized in Sections 10–11.

### 1.3 Task story

An e-commerce picking cell. A storage tote arrives containing one target SKU buried among rigid, irregularly shaped items. The robot must retrieve the target. Blocking items may be staged on a small adjacent tray — the only free surface in the cell — which is deliberately undersized relative to the tote's contents. SKUs are fragile and must not be stacked (this justifies the no-stacking manipulation model). The operative decision each episode: **which subset of blockers to stage**, given that the chosen set must jointly fit on the tray.

---

## 2. Notation, units, and conventions

- All geometry in this document is given in **centimeters**; the PyBullet implementation uses **meters** (divide by 100). The world frame has the floor at z = 0, x–y the horizontal plane.
- A **footprint** is a simple 2D polygon (vertex list, CCW). All objects are right prisms: footprint extruded to a uniform height.
- **Inflation by r** of a polygon means the Minkowski sum with a disc of radius r (implemented with `shapely.buffer(r, join_style=mitre or round)`; round is required for correctness of separation semantics).
- A **stream call** is a single invocation of any sampler or test stream (Section 5.3). Budgets are counted in stream calls, which is the hardware-independent primary unit; wall-clock is reported secondarily with hardware disclosed.
- **Seeding:** every stochastic component (shape library, generator, scrambler, streams, planner tie-breaking) takes an explicit integer seed; an instance's JSON records all seeds (Section 8.7). Reproducibility is a hard requirement.
- Authoritative geometry is computed in **2D exact geometry (Shapely ≥ 2.0)**; PyBullet is the execution/rendering layer and, in the realism tier only, a physics validator. This architectural split is deliberate (Section 12.1): it makes feasibility a deterministic geometric fact (required by C2) rather than a contact-simulation outcome.

---

## 3. Parameter summary table

All defaults below are the core-tier operating point; starred (*) parameters are difficulty dials subject to the Section-10 pilot.

| ID | Parameter | Default | Notes |
|---|---|---|---|
| P1 | Tote interior (x × y) | 40 × 30 cm | walls 1.5 cm thick, 12 cm high, open top |
| P2 | Tray interior (x × y) | 18 × 14 cm | lip 1.0 cm thick, 2 cm high |
| P3 | Tote–tray gap | 6 cm | tray interior AABB at [46, 64] × [2, 16] in world cm if tote interior is [0,40] × [0,30] |
| P4 | Object height | 6 cm | uniform, all objects |
| P5 | Object footprint area | 25–80 cm² | concave polygons, 8–14 vertices, ≥ 1 reflex vertex |
| P6 | Objects per instance (N) | 9–12 | includes target |
| P7* | Witness subset size k | 4 or 5 | |
| P8* | Tray slack σ | 1.15–1.30 | usable tray area ÷ Σ witness footprint areas |
| P9* | Plant margin δ | 2.4 cm | pilot grid {1.8, 2.4, 3.0}; decomposition in §8.2 |
| P10 | Scramble radius r_s | δ/4 | position perturbation; rotation perturbation ± 4° |
| P11 | Finger width w_f | 1.5 cm | tangential extent of each finger |
| P12 | Finger thickness t_f | 1.0 cm | normal extent of each finger |
| P13 | Finger clearance c_f | 0.5 cm | per side, added to finger footprint for column checks |
| P14 | Max gripper aperture | 14 cm | min aperture 0.5 cm |
| P15 | Grasp descent height z_g | 3 cm | finger tips descend to z = 3 cm (object mid-height) |
| P16 | Carry height z_c | 15 cm | object bottom during transfer; clears 12 cm tote walls |
| P17 | Antipodal angle tolerance | 10° | between anti-parallel edge normals |
| P18 | Refinement budget B | 300 stream calls | per skeleton |
| P19 | Per-step retry cap t | 10 | before backjumping (§9.2) |
| P20* | Sampler strength m_p | 15 candidate poses/call | pilot grid {5, 15, 40} |
| P21 | Nester rotation grid | 5° | offline exact solver |
| P22 | Candidate count per instance | 20–40 | enumerator target (§6.2) |
| P23 | Feasible fraction | 10–20% | of enumerated candidates |
| P24 | Decoy count | 3–6 | engineered infeasible candidates (§8.4) |
| P25 | Witness verification | ≥ 4/5 seeds refine within B | §8.6 |
| P26 | MI leakage threshold τ | 0.10 bits | §8.5 audit A4 |

---

## 4. Environment specification

### 4.1 Workspace geometry

Two fixed rigid fixtures on a flat floor (z = 0):

1. **Tote.** Interior floor 40 × 30 cm; four walls of thickness 1.5 cm and height 12 cm; open top. Place the interior AABB at world [0, 40] × [0, 30] cm.
2. **Tray.** Interior floor 18 × 14 cm; lip of thickness 1.0 cm and height 2 cm on all four sides; open top. Interior AABB at [46, 64] × [2, 16] cm (6 cm gap from the tote's outer wall). The **usable tray region** for placement is the interior polygon eroded by δ/2 (wall-separation semantics, §7.3); its area at δ = 2.4 is ≈ (18 − 2.4) × (14 − 2.4) ≈ 181 cm² — use the eroded area, not 252 cm², when computing tray slack σ.

Both fixtures are static bodies in PyBullet. There is no other free surface: the floor outside the fixtures is declared out-of-workspace for placements (the streams never sample it), which is what makes the tray the bottleneck (C3).

### 4.2 Object shape library

Objects are rigid right prisms of uniform height 6 cm whose footprints come from a **procedurally generated concave-polygon library**, published with the benchmark as a JSON file of vertex lists plus the generating seed.

**Generation procedure (per shape):**

```
function generate_shape(seed):
    rng = RNG(seed)
    n_vert = rng.integers(8, 15)                     # 8–14 vertices
    angles = sorted 8–14 samples from [0, 2π) with min gap 0.15 rad
    radii  = rng.uniform(r_min, r_max) per angle      # star-shaped base polygon
    poly   = polygon from polar samples
    # enforce concavity: push 1–3 randomly chosen vertices inward
    for j in rng.choice(vertices, size=rng.integers(1,4)):
        radii[j] *= rng.uniform(0.35, 0.6)
    poly = rebuild polygon; assert poly.is_simple
    assert count_reflex_vertices(poly) >= 1           # else retry
    poly = scale poly to target area A ~ rng.uniform(25, 80) cm²
    assert min_edge_length(poly) >= 1.0 cm            # avoids degenerate grasp edges
    assert poly has >= 1 antipodal edge pair per §4.3.2  # graspable in isolation; else retry
    return poly
```

Rationale (C1/C5): concavity makes subset packability hinge on *shape complementarity* — whether these particular shapes interlock — which is irreducibly joint. Convex footprints must be rejected because their packability is largely determined by low-order summaries (areas, bounding boxes) and greedy bottom-left insertion is near-optimal on convex instances, which would let cheap attacks succeed and void the benchmark. Rigidity (no deformables/articulation) keeps `packs(S, tray)` a deterministic geometric fact, which C2's bimodal labels require.

Library size: ≥ 500 shapes for the training distribution, plus a disjoint held-out set of ≥ 100 shapes generated with different seeds for the D4 held-out-shape generalization axis. One shape per instance is designated the **target** (distinct color and ID; the problem is fully observable, so identification is free and plays no role).

### 4.3 Manipulation model (core tier): SE(2) × lift

The robot is an abstract floating parallel-jaw gripper. This abstraction is standard in the rearrangement literature and is a *stated* scope limitation of the core tier; a realism tier with a Franka/UR5 arm is specified in §12.4 but is not required for the benchmark to function.

#### 4.3.1 Motion model

A transfer is always: **lift** the grasped object vertically to carry height (object bottom at z_c = 15 cm), **translate** in the plane, **lower** vertically to the placement pose, **release**. Because z_c exceeds the tote wall height (12 cm) plus margin, and all objects are 6 cm tall standing on the floor, transfers at carry height are collision-free *by construction*. This is deliberate (C2): motion feasibility is trivial, so all refinement difficulty concentrates in grasps and placements. The motion "stream" is therefore a constant-true test that still counts as one stream call (for uniform accounting).

#### 4.3.2 Grasps: antipodal edge pairs

A top-down parallel-jaw grasp of object o at footprint pose q is parameterized by an ordered pair of footprint edges (e_a, e_b) and a scalar s ∈ [0,1] along their overlapping projection:

- **Antipodal pair enumeration.** For every pair of edges of o's footprint whose outward normals are anti-parallel within tolerance ±10° (P17), compute the separation d(e_a, e_b) along the mean normal direction and the overlap interval of their projections onto the mean tangent direction. The pair is admissible iff 0.5 cm ≤ d ≤ 14 cm (aperture limits, P14) and the overlap interval length ≥ w_f + 2 c_f = 2.5 cm.
- **Grasp sample.** Choose an admissible pair and s (the grasp point along the overlap). This defines two **finger columns**.

#### 4.3.3 Finger columns and the graspability test

Each finger sweeps a vertical rectangular prism while descending. Its footprint is a rectangle of extent (w_f + 2 c_f) × (t_f + 2 c_f) = 2.5 × 2.0 cm, placed immediately outside the corresponding face at the grasp point; its vertical extent is z ∈ [z_g, z_top] with z_g = 3 cm and z_top above everything (25 cm).

**Column check (2D reduction).** An entity blocks a finger column iff its footprint intersects the column rectangle *and* its vertical extent overlaps [z_g, z_top]. Objects (z ∈ [0,6]) overlap [3,25] in [3,6] → objects block columns via pure 2D footprint intersection. Tote walls (z ∈ [0,12]) likewise block in 2D. The tray lip (z ∈ [0,2]) does **not** overlap [3,25] → the lip never blocks fingers. Hence graspability is an exact 2D computation:

```
function graspable(o, q, scene) -> list of (pair, s) with free columns:
    for each admissible antipodal pair of o at pose q:
        for candidate s values:
            build the two column rectangles in world frame
            if neither rectangle intersects any other object footprint,
               any tote wall polygon, or the tray lip-excluded regions:
                yield (pair, s)
```

This single mechanism produces the domain's occlusion structure: a target surrounded by neighbors has all its column rectangles blocked and is **ungraspable** until blockers are removed. It also generates the finger-clearance requirement on tray nests: a placed object's neighbors must leave its columns free if it ever needed re-grasping — the plant's margin budget covers this (§8.2), and placement order within a feasible subset becomes irrelevant (§6.3).

#### 4.3.4 Placement and the no-stacking rule

`place` lowers the held object to z = 0 (object base on the tote floor or tray floor). **Stacking is excluded at the stream level**: the tray and tote placement samplers only emit poses whose full base polygon is supported by the fixture floor (footprint ⊆ usable interior region), and there is no `place-on-object` action in the domain. Story justification: fragile SKUs must not be stacked.

Rationale (C3, C2): if stacking were legal, "pile everything on the tray" would defeat the undersized-tray bottleneck and reintroduce conservative plans; if stacking were legal-but-unstable, feasibility would become a stochastic stability question, breaking bimodal labels and forcing an expensive 3D stability checker. Excluding it by the manipulation model, stated upfront, is the clean resolution. Consequence: under quasi-static kinematic execution, flat rigid prisms on flat lipped surfaces cannot tumble, so `in-tote(o) ∨ on-tray(o) ∨ holding(o)` is invariant and **no end-of-episode "did anything fall off" check is needed in the core tier**. (The realism tier adds a settling check; §12.4.)

The gripper holds at most one object (holding capacity 1). Pushing, dragging, and multi-object grasps do not exist.

---

## 5. Symbolic model and streams (PDDLStream-style)

### 5.1 Domain

```
;; Types: obj. Constants: target ∈ obj.
;; Static per-instance: shape(o), initial pose(o).

(:predicates (holding ?o) (handempty) (in-tote ?o) (on-tray ?o) (extracted ?target))

(:action pick
  :parameters (?o ?g)                       ; g from grasp stream
  :precondition (and (handempty) (in-tote ?o) (grasp-valid ?o ?g))   ; test stream
  :effect (and (holding ?o) (not (in-tote ?o)) (not (handempty))))

(:action place-tray
  :parameters (?o ?p)                       ; p from tray placement stream
  :precondition (and (holding ?o) (tray-pose-valid ?o ?p))            ; test stream
  :effect (and (on-tray ?o) (handempty) (not (holding ?o))))

(:action place-tote
  :parameters (?o ?p)                       ; p from tote placement stream
  :precondition (and (holding ?o) (tote-pose-valid ?o ?p))
  :effect (and (in-tote ?o) (handempty) (not (holding ?o))))

(:action retrieve
  :parameters (?g)
  :precondition (and (handempty) (in-tote target) (grasp-valid target ?g))
  :effect (and (extracted target) (not (in-tote target))))

;; Goal: (extracted target)
```

Two rules with rationale:

1. **`clear-grasp(target)` is a test stream, never a static literal.** Do *not* precompute `blocks-grasp(o, target)` literals into the initial state. If blocking were symbolic and pairwise, (a) a classical planner would derive the clearing set by set-cover over literals, re-importing exactly the low-order structure C1 forbids, and (b) candidate diversity would collapse to the single minimal cover, starving the enumerator. Instead the symbolic planner reasons *optimistically* — removing various subsets *might* free a grasp — and grasp existence is checked geometrically at refinement time by `grasp-valid`. Geometry stays with the streams; the symbolic layer stays geometry-blind.
2. **`place-tote` is legal but geometrically dead in the core tier.** The generator guarantees (audits A2/A5, §8.5) that in-tote repositioning cannot help, but the action is not removed from the domain, so solvers are not artificially restricted — an important fairness point for G4.

### 5.2 Streams

| Stream | Signature | Semantics |
|---|---|---|
| `sample-grasp(o)` | → g = (edge pair, s) | seeded sampler over the admissible antipodal pairs of o at its *current* pose; yields one grasp candidate per call |
| `grasp-valid(o, g)` | → bool | column check of §4.3.3 against the current scene |
| `sample-tray-pose(o)` | → p = (x, y, θ) | **compaction-biased** placement sampler over the usable tray region (§5.4) |
| `tray-pose-valid(o, p)` | → bool | footprint at p ⊆ usable tray region, and no intersection with already-placed tray objects, and the *approach columns* of the placing grasp are free at p (fingers must be able to descend and open at the placement) |
| `sample-tote-pose(o)` | → p | uniform over tote free space (kept weak on purpose; it only matters for the dead action) |
| `tote-pose-valid(o, p)` | → bool | analogous validity in the tote |
| `motion-ok(...)` | → true | constant (§4.3.1), still accounted |

Each invocation of any row = **one stream call**. All samplers consume the instance-level stream seed plus a per-skeleton nonce so that refinement attempts are reproducible.

### 5.3 Stream-call accounting

The primary cost metric everywhere in the benchmark (budgets, G2, D2, checker-in-loop comparisons) is total stream calls; report wall-clock secondarily with hardware stated. The checker-in-loop baseline's nester invocations are accounted separately in wall-clock *and* converted to equivalent-stream-call units by measuring the mean wall-clock of one stream call on the same hardware; report both.

### 5.4 The compaction-biased tray placement sampler (critical component)

A uniform rejection sampler over (x, y, θ) fails this benchmark: at tray slack 1.15–1.30 over concave shapes, uniformly sampled early placements waste space and even the *planted* subset would rarely refine, violating C2's ceiling and killing Gate 1. The shared refiner therefore uses a **compaction-biased** sampler — cheap, incomplete, randomized, and identical for every method under evaluation (the benchmark measures skeleton selection *given* this fixed refiner):

```
function sample_tray_pose(o, placed, rng, m_p):
    # m_p = sampler strength (candidate poses per call), a pilot dial
    θ ~ rng: uniform from a 15° grid + jitter U(−7.5°, +7.5°)
    R = usable tray region eroded by nothing (validity handles separation)
    candidates = []
    repeat m_p times:
        draw a proposal:
          with prob 0.7: "contact proposal" — sample a pose on the boundary of the
              free configuration space of o (at θ) w.r.t. placed ∪ tray walls,
              i.e., touching (within numeric ε) an already-placed object or the wall.
              Implementation: compute NFP(o(θ), placed_i) boundaries (§7.1) and the
              inner-fit polygon of o(θ) in R; sample a point on the union of these
              boundaries, offset outward by stream tolerance ε_s = 0.15 cm.
          with prob 0.3: uniform pose in the free space, then translate toward the
              nearest contact along a random direction until first contact, back off ε_s.
        if footprint(o, pose) is collision-free: candidates.append(pose)
    if candidates empty: return FAIL                       # counts as the call's result
    return candidate minimizing (y, then x) with Gumbel noise of temperature 0.3 cm
           # randomized bottom-left preference
```

Two properties must hold and are what the pilot verifies: (i) **strong enough** that planted (δ-margined) nests are found within budget B with per-seed success ≥ 0.9 when the subset is feasible; (ii) **not a cheap certificate** — on infeasible subsets the sampler can only fail-to-find, which costs the full budget B. That expensive failure is not a bug; it *is* C5's costly-rejection mechanism. What C5 forbids is a *fast* reliable check (≤ tens of milliseconds), and those are the heuristics attacked in §10.3.

---

## 6. Skeleton space, candidate enumeration, and the latent statistic

### 6.1 Skeleton = blocker subset

Every useful plan has the form

```
pick(o₁,·); place-tray(o₁,·); … ; pick(o_k,·); place-tray(o_k,·); retrieve(target,·)
```

for some subset S = {o₁,…,o_k} of non-target objects. The benchmark identifies skeletons with **subsets** S; ordering within S is feasibility-irrelevant by design and audit (§6.3), and the enumerator emits one canonical order (ascending object ID) per subset.

**Latent feasibility statistic:** `feasible(S) ⇔ clears(S) ∧ packs(S, tray)`. The generator guarantees (audit A1) that *every enumerated candidate clears*, so the discriminating statistic is `packs(S, tray)` alone — the order-k continuous nesting question (C1).

### 6.2 Candidate enumeration (the "diverse planner order")

Candidates arise from the target's grasp-corridor structure:

```
function enumerate_candidates(instance):
    corridors = all admissible antipodal grasps of the target at its initial pose,
                clustered by (edge pair, s-interval) into maximal corridor segments
    for each corridor c:
        blockers(c) = { o ≠ target : footprint(o) intersects either finger-column
                        rectangle of c }            # walls may also block; corridors
                                                     # blocked by walls are discarded
    base = unique minimal sets among { blockers(c) }   # remove supersets
    supersets = for each b in base, b ∪ {one adjacent non-member object},
                until total count reaches the 20–40 target (P22)
    return dedupe(base ∪ supersets), each as a canonical-order skeleton
```

The **uninformed baseline order** over candidates is: ascending |S|, ties broken by a seeded random permutation. This ordering is fixed, published, and used for the D1 diagnostic (enumeration rank of the first feasible candidate). Rationale: minimal-length-first is what FORBID-style diverse enumeration produces, and randomized tie-breaking prevents accidental correlation between object IDs and feasibility.

All candidates clear by construction (removing `blockers(c)` frees corridor c), but audit A1 (§8.5) re-verifies each with margin, because supersets and scrambling can perturb the corridor structure.

### 6.3 Order-irrelevance within feasible subsets

Claim: if S is feasible (a δ-inflated nest exists) then any staging order refines, because top-down insertion into a valid nest requires only that each placement's approach columns are free at placement time, and the δ budget includes finger clearance for every object in the nest (§8.2). Audit A6 verifies empirically: for each instance, refine the witness under 3 random orders × 2 seeds; all must succeed. If order-dependence is ever detected, the instance regenerates (and a systematic failure here indicates δ is under-budgeted — a pilot signal, not a patch-in-place).

---

## 7. Offline nesting solver (exact checker) and the feasibility label rule

### 7.1 Machinery: no-fit polygons

"Nesting" (from garment/sheet-metal cutting) is 2D irregular packing: place polygons in a container without overlap, rotations allowed. The standard exact-in-practice machinery:

- **NFP(A, B):** the no-fit polygon of B relative to A — the locus of B's reference point at which A and B touch; B's reference point strictly inside NFP ⇔ overlap. Compute via Minkowski sum A ⊕ (−B): decompose both polygons into convex parts (constrained triangulation or Hertel–Mehlhorn), take pairwise convex Minkowski sums (linear-time per pair), union with Shapely.
- **IFP(A, C):** inner-fit polygon — the locus of A's reference point keeping A inside container C; computed analogously against C's boundary.

### 7.2 Solver

```
function nest(shapes, container, rot_grid=5°, time_cap, node_cap) -> nest | INFEASIBLE | TIMEOUT:
    order heuristics: try shapes by descending area (plus 2 random restarts of order)
    depth-first search over placements:
        at depth i, for each rotation θ in rot_grid (+ per-restart random offset):
            candidate positions = vertices and edge-midpoints of the boundary of
                IFP(shape_i(θ), container) ⊖ ∪_j NFP(shape_i(θ), placed_j)
            sort candidates bottom-left; recurse
        backtrack on exhaustion
    return first complete placement, or INFEASIBLE if the search space is exhausted,
    or TIMEOUT at time_cap / node_cap
```

For k ≤ 6 shapes this is effectively exact at seconds-to-minutes per subset — affordable offline (labels, audits) and *deliberately* unaffordable at plan time, which is the cost asymmetry C5 exploits. Two run modes: **exact mode** (generator/labels: generous caps, TIMEOUT treated as "regenerate or escalate caps") and **anytime mode** (checker-in-loop baseline: hard per-call cap, e.g., 2 s or 5×10⁴ nodes, returning FOUND / NOT-YET).

### 7.3 The bimodal label rule (load-bearing for C2)

For a candidate subset S:

- **feasible(S)** ⇔ a nest of the **δ/2-inflated** shapes of S exists in the tray interior. (Inflating each shape by δ/2 guarantees ≥ δ pairwise separation and ≥ δ/2 wall separation in the real tray. Note the direction: inflate the *objects* / erode the container — never dilate the container, which would remove margin instead of adding it.)
- **infeasible(S)** ⇔ no nest exists even at **zero** inflation.
- **marginal** (nest exists at 0 but not at δ/2 inflation) ⇒ the instance is **regenerated** (or the offending decoy resampled). Marginal candidates are exactly the ones whose refinement outcome would depend on sampler luck, which would poison labels and violate C2.

This checker-level rule *enforces* bimodality rather than merely auditing it; the sampling audit (§8.6) then confirms the stream-level consequence (≥ 0.9 / ≤ 0.1 refinement rates).

---

## 8. Planted instance generator

### 8.1 Overview

Instances are generated **backward** from a solution ("planted generation"): sample a feasible witness subset and its geometric certificate first, then construct the scene around it. Consequences: every instance ships with a solvability certificate (Gate 1 support), positive examples are free (no baseline planner needed to discover solutions), and the oracle bracket (a planner told the witness) is available by construction. The known risk of planting — statistical fingerprints that let a model detect the plant instead of reasoning — is countered by scrambling plus explicit leakage audits (A3/A4).

```
function generate_instance(seed) -> instance | RESTART:
    rng = RNG(seed)
    1. plant the witness nest                         (§8.2)
    2. build the tote occlusion structure              (§8.3)
    3. engineer decoys + fill distractors (repair loop)(§8.4)
    4. scramble                                        (§8.6)
    5. run audits A1–A6; regenerate on any failure     (§8.5, §8.6)
    6. enumerate candidates, label all via exact nester(§6.2, §7.3)
    7. emit instance JSON                              (§8.7)
```

Expect a rejection-and-repair style generator; a per-seed RESTART rate up to ~70% at tight dials is acceptable and should be logged per audit (the pilot uses these rates to map the feasible dial window).

### 8.2 Step 1 — plant the witness nest

Choose k ∈ {4,5} (P7). Draw candidate witness shapes from the library subject to the slack constraint: Σ areas ∈ [A_tray_usable/σ_max, A_tray_usable/σ_min] with σ ∈ [1.15, 1.30] (P8) and A_tray_usable the δ/2-eroded tray interior area (§4.1). Run the exact nester on the **δ/2-inflated** witness shapes; if INFEASIBLE/TIMEOUT, resample shapes (up to 50 tries, then RESTART). Record the nest (the planted tray poses) as the witness certificate.

**Margin budget δ (default 2.4 cm), decomposition and rationale:**

| Consumer | Allocation | Why |
|---|---|---|
| Finger clearance | ~1.0 cm (2 × c_f) | approach columns at each planted pose must be free so any placement order works (§6.3) |
| Scramble radius | ≤ 2 r_s = δ/2 | post-plant perturbations (§8.6) must not create collisions or kill the nest |
| Stream tolerance ε_s | ~0.3 cm | sampler pose precision + simulator contact slop; a pose within ε_s of a valid nest must still validate |

δ is a pilot dial (P9): too small → planted subsets don't survive scrambling/sampling (C2 ceiling breaks); too large → the tray becomes loose and heuristics start certifying correctly (C5 floor breaks). The three-dial pilot (§10.2) locates the window.

### 8.3 Step 2 — build the tote occlusion structure

Place the target near the tote center (jittered). Choose a designated corridor direction c* (a random admissible antipodal grasp of the target). Place the k witness objects such that:

- each witness object's footprint intersects at least one finger-column rectangle of *every* grasp corridor in a chosen "primary corridor cluster" containing c* — operationally: iterate `while ∃ free target grasp: place the next witness object (δ-separated from everything) to intersect the columns of the freest remaining corridor`;
- after placing all k, **every** target corridor is blocked by ≥ 1 object (distractors added in Step 3 may take over blocking of secondary corridors);
- removing the full witness set frees corridor c* **with margin**: re-run graspability with columns inflated by 0.5 cm; must succeed.

The intended relationship, verified rather than assumed: `blockers(c*) = witness` after Step 3 completes.

### 8.4 Step 3 — decoys and distractors (generate-and-repair loop)

**Decoys** are the anti-certificate device (C5) and the anti-marginals device (combination dependence): candidate subsets that also clear the target, **pass the summed-area bound** (Σ areas ≤ A_tray_usable — even ≤ the witness's Σ), yet **provably fail to nest** (exact nester returns INFEASIBLE at zero inflation). Their existence guarantees that no linear/aggregate statistic separates feasible from infeasible.

```
repair loop:
    place remaining distractor objects (total N ∈ [9,12]) around the occlusion
    structure, δ-separated, biased to (i) block secondary corridors of the target and
    (ii) sit adjacent to witness objects so that corridor blocker sets overlap heavily
    recompute corridors and blocker sets; enumerate candidates (§6.2)
    label candidates with the exact nester (label rule §7.3)
    check acceptance conditions:
        A-count: 20–40 candidates                                   (P22)
        A-feas:  10–20% feasible                                    (P23)
        A-decoy: ≥ 3 infeasible candidates pass the summed-area bound (P24)
        A-olap:  every witness object appears in ≥ 1 infeasible candidate;
                 every object of every decoy appears in ≥ 1 other candidate
        A-marg:  zero marginal candidates (§7.3)
    if a condition fails: local repair (move/swap one distractor to join/leave a
    corridor's blocker set; resample one decoy-implicated shape) and re-check;
    after R=30 repairs, RESTART
```

**Distractor fill condition (kills in-tote shortcuts, C3):** continue adding/adjusting distractors until no free pocket in the tote admits the smallest object's δ/2-inflated footprint at any rotation (checked with IFP against tote free space). Combined with the structural fact that every candidate has |S| ≥ 4 while holding capacity is 1, proactive stashing and single-object repositioning cannot clear the target; audit A5 verifies the residual possibility empirically.

### 8.5 Audits (run after scrambling; any failure ⇒ regenerate)

- **A1 — Clearing with margin.** For every enumerated candidate S: remove S geometrically; the target must have ≥ 1 grasp whose columns, inflated by 0.5 cm, are free. (Guarantees `clears` never discriminates among candidates; packing alone does.)
- **A2 — No initial pocket.** The distractor fill condition of §8.4 holds after scrambling.
- **A3 — Plant-detection attack (conditional form, G3).** Train the Tier-0 model suite (§10.3's learned tier: pairwise-factorized models over per-object and pairwise geometric features, plus a triplet-feature model) to predict (i) *which candidate is the witness* among each instance's candidate set and (ii) object-level witness membership *among clearing-candidate objects*. Required: performance indistinguishable from the symbolically-informed baseline rate (e.g., uniform over candidates for (i)). This is dataset-level (run per generation batch, e.g., every 500 instances), not per-instance. **Important:** the attack population is *candidates*, not all objects — witness objects are necessarily near the target, so an unconditional attack would "succeed" spuriously on distance-to-target without any true leakage.
- **A4 — Combination dependence.** Across each instance's candidate set with labels Y: for every object o, the empirical mutual information I([o ∈ S]; Y) ≤ τ = 0.10 bits, and no single-object indicator is a perfect predictor. (Prevents "avoid the bulky one" single-failure generalization.)
- **A5 — No shuffle shortcut (empirical).** Run the uninformed baseline planner (§9.3) with `place-tote` enabled at generous budget on a 5% sample of instances; verify every solution found stages a clearing subset to the tray and none uses `place-tote` productively.
- **A6 — Order irrelevance.** §6.3.

### 8.6 Steps 4–5 — scramble and verify

**Scramble:** perturb every object's tote pose by a uniform offset of magnitude ≤ r_s = δ/4 and rotation ≤ ±4°, rejection-sampled to preserve δ/2 minimum separations. Purpose (G3): destroy generator fingerprints — suspiciously regular spacings, corridor geometry that echoes the construction order — without destroying the witness (the margin budget was sized for exactly this).

**Witness verification:** refine the witness skeleton with the shared refiner (§9.2) under 5 independent stream seeds; require ≥ 4/5 successes within B = 300 stream calls (P25). Then re-run audits A1, A2 and re-label candidates (scrambling can flip a near-boundary decoy — the label rule's regeneration clause handles it).

### 8.7 Instance JSON schema

```json
{
  "instance_id": "ttd-000123",
  "generator_version": "1.0",
  "seeds": {"generator": 123, "scramble": 456, "stream_base": 789},
  "dials": {"k": 5, "slack": 1.22, "delta_cm": 2.4, "sampler_strength": 15},
  "fixtures": {"tote_interior": [[0,0],[40,0],[40,30],[0,30]], "wall_h": 12,
               "tray_interior": [[46,2],[64,2],[64,16],[46,16]], "lip_h": 2},
  "objects": [
    {"id": "obj_00", "shape_id": "lib_0412", "vertices_cm": [[..],..],
     "pose": {"x": 12.3, "y": 18.1, "theta_rad": 0.62}, "is_target": false},
    ...
  ],
  "target_id": "obj_07",
  "witness": {"subset": ["obj_01","obj_03","obj_04","obj_09"],
              "tray_nest": [{"id": "obj_01", "x": 48.2, "y": 4.1, "theta_rad": 1.9}, ...],
              "cleared_corridor": {"edge_pair": [2,6], "s": 0.41}},
  "candidates": [
    {"subset": ["obj_01","obj_03","obj_04","obj_09"], "label": "feasible",
     "nester_result": "nest_at_delta"},
    {"subset": ["obj_01","obj_02","obj_04","obj_09"], "label": "infeasible",
     "nester_result": "no_nest_at_zero", "passes_area_bound": true, "is_decoy": true},
    ...
  ],
  "audit_results": {"A1": true, "A2": true, "A4_max_MI_bits": 0.06,
                    "A5": "sampled_pass", "A6": true,
                    "witness_verification": {"successes": 5, "seeds": 5, "budget": 300}}
}
```

Splits: train / val / test / **held-out-shape test** (D4, disjoint shape library) / **out-of-generator test** (G3: a small set, e.g., 100 instances, produced by rejection-sampling natural clutter scenes through a generous-budget solver — biased easy by construction, so only method *rankings*, never absolute gaps, are claimed on it; state this caveat wherever it is reported).

---

## 9. Planner and refiner

### 9.1 Architecture

All planners share: the candidate enumerator (§6.2), the streams (§5.2) including the compaction-biased tray sampler, the refiner (§9.2), and stream-call accounting. They differ only in the **order/policy over candidates** — which is the point: the benchmark isolates skeleton selection.

### 9.2 Refiner (per skeleton, budget B)

Sequential refinement with bounded backjumping:

```
function refine(skeleton = [pick(o₁),place-tray(o₁),…,retrieve], seed, B):
    calls = 0; bindings = []          # chosen grasps/poses per step
    step = 0
    per_step_tries = zeros
    while calls < B:
        a = skeleton[step]
        try to bind a:
            pick(o):        g ~ sample-grasp(o); test grasp-valid(o,g)        # 2 calls
            place-tray(o):  p ~ sample-tray-pose(o); test tray-pose-valid    # 2 calls
            retrieve:       g ~ sample-grasp(target); test grasp-valid       # 2 calls
            motion-ok                                                        # 1 call
        calls += as incurred
        if bound: apply geometric effect (object leaves tote / lands at p);
                  bindings.append(...); step += 1
                  if step == len(skeleton): return SUCCESS(bindings, calls)
        else:
            per_step_tries[step] += 1
            if per_step_tries[step] >= t (=10):        # P19
                if step == 0: continue                  # keep trying step 0 until budget
                undo step−1's effect; discard its binding
                per_step_tries[step] = 0; step −= 1     # backjump: re-place the previous
                                                        # object elsewhere, freeing space
    return FAILURE(calls = B)
```

Notes. (i) Backjumping is what gives the refiner a real chance on feasible-but-snug subsets: a bad early tray placement gets revised. (ii) On **infeasible** subsets the characteristic trace is: the first k−1 tray placements succeed, the k-th fails repeatedly, backjumping shuffles earlier placements, and the budget exhausts — the **late-failure** cost profile C5 requires. Diagnostic D2 records, per rejected skeleton, the deepest step reached and total calls. (iii) The refiner never calls the offline nester.

### 9.3 The planner variants (all required)

1. **Uninformed baseline (SeSamE-style, Gate 1/2 reference).** Iterate candidates in the published diverse order (§6.2); refine each with budget B; return the first SUCCESS. *Generous budget* (Gate 1): every candidate refined at 3 × B, whole-instance retry at 3 seeds — must solve ≥ 95% of instances. *Practical budget* (Gate 2): B per candidate, single pass — expected profile at the operating point: ~5 failed refinements × ~300 calls + ~50–150 calls on the hit ≈ 1,500–1,700 calls/solve.
2. **Planted oracle (upper bracket).** Reads the witness from the instance JSON and refines only it: ~50–150 calls/solve, ~100% success. The headroom ratio vs. (1) is the benchmark's advertised gap (target 10–15×).
3. **Checker-in-loop (C5 bracket).** Before refining a candidate, call the anytime nester (§7.2) with a hard cap; skip candidates it proves/strongly-suggests infeasible; refine the first survivor. Report stream calls, nester wall-clock, and equivalent-call conversion (§5.3). This baseline is *expected to be strong*; the research gap for any learned method is to approach or beat its per-instance cost by amortization, and to keep working where exact checking would time out (larger k tiers). Pre-register both possible outcomes.
4. **Constant policy (C3 audit).** The single best fixed rule fit on training data (e.g., "stage the k nearest blockers by centroid distance" and best-single-lifted-template variants); must perform near the uninformed baseline, far from the oracle.
5. **Retrieval baseline (D3).** Nearest-neighbor over instances (feature: multiset of shape descriptors + coarse poses); reuse the neighbor's witness subset mapped by shape similarity; refine only that. Must fail — this is the direct anti-memoization test.
6. **(G4, separate work package) incumbent re-implementations.** PIGINet-style: 512² top-down render + per-object crops + literals, trained/tuned on the training split, reranking the same candidate set. LAZY-style: GAT policy over the scene graph (poses, extents, pairwise offsets, predicates) with online statistics, driving candidate order. Both validated first on sanity tasks matching their home turf. These are evaluation subjects, not infrastructure, and can be implemented after the environment ships.

---

## 10. Difficulty dials, operating point, and the C6 three-dial pilot

### 10.1 Dials

Primary (pilot grid): tray slack σ ∈ {1.15, 1.20, 1.25, 1.30} × plant margin δ ∈ {1.8, 2.4, 3.0} cm × sampler strength m_p ∈ {5, 15, 40}. Secondary (fixed during the pilot, escalation levers later per the pre-registered protocol): k, concavity strength, decoy count, N.

### 10.2 Pilot protocol (run before any full dataset build)

For each of the 36 grid cells: generate 200 instances (recording RESTART rates per audit), then measure:

- **(a) C2 ceiling:** planted-witness refinement success rate under the shared refiner, 5 seeds each — **pass ⇔ ≥ 90%** of instances have ≥ 4/5 seed successes.
- **(b) C5 floor:** run the heuristic-certificate suite (§10.3) on all labeled candidates — **pass ⇔ every heuristic's balanced accuracy ≤ 0.60.** (Balanced accuracy, not raw accuracy: with a 10–20% feasible base rate, the trivial "always infeasible" classifier scores 80–90% raw accuracy; balanced accuracy pins chance at 0.50.)
- **(c) G1+G2:** uninformed baseline at generous budget solves ≥ 95%; at practical budget, failed-refinements-per-success ∈ [5, 10].

**Pilot pass ⇔ some cell satisfies (a) ∧ (b) ∧ (c) simultaneously.** That cell becomes the operating point. If no cell passes: first extend the grid one step along the failing direction (e.g., larger m_p if (a) fails everywhere; tighter σ if (b) fails everywhere); if the extended grid also fails, TTD's packing coupling is declared unworkable and the pre-registered fallback (reachability/occlusion-geometry coupling) replaces it. Do not silently weaken conditions (a)–(c).

### 10.3 Attack suites (referenced by A3, pilot (b), and D5)

**Heuristic certificates (hand-coded, must fail per pilot (b)):** H1 — summed-area bound (Σ areas ≤ A_tray_usable); H2 — one-shot greedy bottom-left insertion, 30° rotation grid, no backtracking, descending-area order; H3 — bounding-box strip-packing bound; H4 — H2 with 3 random restarts (a slightly stronger greedy; include it so the claim is robust).

**Tier-0 learned attack (C1/A3, must sit near chance on feasibility and on plant detection):** logistic/GBDT models over features that factorize as Σᵢ f(oᵢ) + Σᵢⱼ g(oᵢ,oⱼ) + simple aggregates — per-shape descriptors (area, perimeter, convexity defect, aspect), pairwise descriptors (NFP area between the pair, complementarity score), aggregates (Σ area, max width, count) — plus an explicit **triplet-feature** variant (all C(k,3) triple-nesting bits, precomputed) to close the "k=4 is only order-4" loophole from below. Fixed training budget: the training split's labels, standard hyperparameter search, no feature learning.

**Mid-tier model (C4 bracket, should land strictly between Tier-0 and oracle):** a set transformer over per-shape encodings of the candidate subset + tray descriptor, trained on candidate labels. Its success or failure decides the paper's framing (representational vs. integration gap) — pre-registered either way — but the benchmark's validity does not depend on which occurs.

---

## 11. Metrics, gates, and diagnostics (reporting spec)

Per method, per split, report: solve rate; stream calls per solve (mean, median, P90); failed refinements per solve; wall-clock (hardware stated); and the bracket table (uninformed / method / checker-in-loop / oracle). Diagnostics: **D1** — rank of the first feasible candidate in the published diverse order (distribution); **D2** — per-rejection deepest-step and call cost (validates late failure; also the mechanism-pairing diagnostic for LAZY-style online statistics); **D3** — retrieval-baseline solve rate + entropy of witness identity given Tier-0 features (anti-memoization); **D4** — held-out-shape split at fixed N (controlled generalization) and N → 1.5N extrapolation (stress test; report with the explicit caveat that count extrapolation changes intrinsic difficulty); **D5** — per-heuristic balanced accuracy / TPR / FPR on candidate labels. Gates G1–G3 as defined above; G4 reports each incumbent's degradation paired with its predicted mechanism (PIGINet ↔ D1; LAZY ↔ D2).

---

## 12. Software architecture and implementation plan

### 12.1 Two-layer architecture (authoritative 2D core + PyBullet layer)

**All feasibility-relevant geometry — grasp columns, placement validity, nesting, corridors, audits — is computed in the 2D core (Shapely).** PyBullet renders and executes; in the core tier it is *kinematic*: bodies are moved by `resetBasePositionAndOrientation` along interpolated lift–translate–lower trajectories, with `stepSimulation` used only for visual continuity, not for determining outcomes. Rationale: C2 demands deterministic feasibility; concave dynamic bodies in PyBullet require convex-decomposed compound collision shapes whose contact behavior is noisy at millimeter scales, and letting the simulator adjudicate placements would smear the bimodal labels. The realism tier (§12.4) is where physics gets a vote.

**PyBullet body construction for concave prisms:** triangulate each footprint (Shapely `triangulate` constrained to the polygon, or ear clipping), extrude each triangle to a 6 cm convex prism, assemble via `createCollisionShapeArray`/`createVisualShapeArray` + `createMultiBody` as a compound body. The gripper is two finger boxes (1.5 × 1.0 × 8 cm) on a floating base, moved kinematically; aperture animates between max and the grasp separation.

### 12.2 Module breakdown

```
ttd/
  shapes.py        # library generation (§4.2), shape descriptors, held-out split
  geometry.py      # inflation/erosion, convex decomposition, Minkowski, NFP/IFP
  grasps.py        # antipodal enumeration, finger columns, corridor clustering (§4.3, §6.2)
  nesting.py       # exact + anytime nester (§7)
  streams.py       # all streams + call accounting + seeding (§5)
  refine.py        # refiner (§9.2)
  planners.py      # uninformed / oracle / checker-in-loop / constant / retrieval (§9.3)
  enumerate.py     # candidate enumeration + published ordering (§6.2)
  generator.py     # plant, occlusion build, decoy repair loop, scramble, audits (§8)
  attacks.py       # heuristic suite, Tier-0 models, plant-detection attack (§10.3)
  sim.py           # PyBullet scene build, kinematic execution, cameras
  video.py         # recording + overlays (§13)
  pilot.py         # §10.2 grid runner
  metrics.py       # §11 reporting
  schemas/instance.schema.json
```

Dependencies: Python ≥ 3.10, `pybullet`, `shapely ≥ 2.0`, `numpy`, `scikit-learn` (attacks), `ffmpeg` (video). Optional: `triangle` or `mapbox_earcut` (triangulation), `pyclipper` (robust offsets).

### 12.3 Build order and acceptance tests

1. `shapes` + `geometry` + `nesting` — unit tests: known-feasible and known-infeasible hand-made nests; inflation direction test (a nest at δ/2 inflation must imply ≥ δ pairwise clearance measured in the real tray).
2. `grasps` + `streams` + `refine` — test: a hand-built loose scene where every subset refines; verify call accounting and seeding reproducibility (identical seed ⇒ identical trace).
3. `enumerate` + `generator` (without decoy repair) — test: planted witness refines ≥ 4/5 seeds pre-scramble.
4. Full generator with audits — test: 50 instances pass A1–A6; log RESTART causes.
5. `planners` + `metrics` — test: oracle ≈ 100% at ~100 calls; uninformed solves with the late-failure trace visible in D2.
6. `pilot.py` on the 36-cell grid (this is the project's first real experiment and its go/no-go).
7. `attacks.py`, batch A3 audit, dataset build at the passing cell, `video.py`.

### 12.4 Realism tier

Same scenes with a Franka Panda (or UR5) in PyBullet: planting re-verifies arm reachability and top-down approach feasibility at every witness pose (IK + collision along the descent); dynamic execution with friction; a **settling check** after each place (max object displacement < 0.3 cm over 0.5 s of simulation) and a terminal pose audit (`in-tote ∨ on-tray` for all objects — this is where the "did anything tumble" check, unnecessary in the core tier, becomes real). Core-tier labels remain the ground truth; the realism tier reports execution success of core-tier plans.

---

## 13. Video evidence pipeline

Purpose: qualitative evidence for the paper and debugging — visibly *why* wrong subsets are expensive (k−1 clean placements, then repeated failure at the last object) and how planted subsets succeed.

**Cameras.** Two fixed views rendered every frame via `getCameraImage` (do not rely on `STATE_LOGGING_VIDEO_MP4`, which only captures the GUI window): (V1) top-down orthographic-like view framing tote + tray (yaw 0, pitch −89.9°, distance ~0.9 m, target midpoint between fixtures); (V2) oblique view (yaw 45°, pitch −35°, distance 1.1 m). 640 × 480 minimum, 30 fps, encoded with ffmpeg (`libx264`, CRF 20).

**Motion rendering.** Kinematic interpolation: pick = gripper descends over the grasp point (0.5 s), fingers close, lift to z_c (0.5 s); transfer = straight-line planar move at 20 cm/s; place = lower (0.5 s), fingers open, ascend.

**Failure visualization.** Each failed placement sample is shown as a translucent red "ghost" of the object's footprint prism at the rejected pose for 3 frames (spawn a ghost visual-only body, then remove); each failed grasp sample flashes the two finger-column prisms in red. Backjumps are shown by the previously placed object lifting and re-placing. On budget exhaustion, freeze 1 s with a "REFINEMENT FAILED (budget B reached)" overlay.

**Overlays** (post-hoc via ffmpeg `drawtext` from a per-frame JSONL event log, or PyBullet `addUserDebugText`): instance ID; candidate index / total; the candidate subset (object IDs, color-highlighted in-scene); running stream-call counter; per-step tries; outcome banner.

**Deliverables per demonstration instance:** (i) `success_<id>.mp4` — oracle or first-feasible refinement end-to-end, including the retrieve; (ii) `failure_<id>.mp4` — one decoy refinement: k−1 placements succeed, the last object's ghosts accumulate, backjumps occur, budget exhausts; (iii) `session_<id>.mp4` — the uninformed planner's full run as a montage (each candidate compressed to ~8 s: first placement, last success, failure burst, verdict card), ending with the successful candidate at full speed. Record the event log (JSONL: timestamps, stream calls, poses, outcomes) alongside every video for exact reproducibility.

---

## 14. Known risks, fallbacks, and conceded limitations

1. **The three-dial window may be empty** (C2 ceiling vs. C5 floor). This is the project's primary risk; it is resolved by the §10.2 pilot *before* dataset construction, with grid extension rules and the pre-registered fallback (reachability/occlusion coupling) if it fails. Do not weaken pass conditions.
2. **Generator repair loop may thrash** at tight dials (high RESTART rates). Log per-audit failure causes; if A-decoy dominates, enlarge the shape library's high-complementarity region; if A2 dominates, raise N's upper bound to 13.
3. **Planted scope.** The benchmark measures skeleton-selection efficiency on solvable-with-margin instances; it does not characterize behavior on natural, near-infeasible clutter. The out-of-generator test set partially mitigates; only method rankings are claimed there. State this in any publication.
4. **Abstracted kinematics** in the core tier (SE(2)×lift, floating gripper); the realism tier is the partial mitigation.
5. **Enumeration-relative metrics.** D1 and the candidate set depend on the published enumerator; all comparisons must use the shipped enumerator and ordering.
6. **If a fairly-run incumbent closes the gap** after the pre-registered escalation protocol (raise k, tighten σ within Gate-1 limits), the gap claim is falsified and must be reported as such. The benchmark's escalation levers and Gate-1 limits are part of the spec precisely so this outcome is a measurement, not a judgment call.

---

## 15. Quick-start summary for the implementer

Build order: 2D geometric core → nester → streams/refiner → generator with audits → planners → **pilot (go/no-go)** → attacks + dataset → video. The single most delicate components, in order: the compaction-biased placement sampler (§5.4 — the C2 ceiling lives or dies here), the decoy repair loop (§8.4 — C5's floor), and the label rule's inflation direction (§7.3 — inflate objects, never the container). Everything downstream (planners, metrics, videos) is conventional once those three are correct.
