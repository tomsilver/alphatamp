# DD2D — Drawer Decluttering in 2D: Implementation Specification

**Version:** 3.0 (post adversarial review; changes from v2.0 marked ⟦R⟧ with the motivating review item M1–M8/m1–m7)
**Stack:** Python ≥ 3.10, Shapely ≥ 2.0 (GEOS version pinned in `requirements.txt` — the bitwise-reproducibility claim depends on it ⟦R:m6⟧), NumPy, scikit-learn (attack models only), Matplotlib + ffmpeg (rendering/video), PDDLStream + Fast Downward (off-the-shelf planner tier)
**Audience:** an implementer with TAMP familiarity but no access to prior design discussions. This document is self-contained: it explains the research purpose and design philosophy, then specifies the world model, the PDDLStream domain encoding, the forward instance generator, ground-truth labeling, the two-tier planner evaluation protocol, the buffer-slack sweep experiment, and the animation pipeline.
**Scale:** the controlled (Tier-2) pipeline is implementable with first planner-vs-oracle numbers in roughly one day of focused work (with an AI coding assistant); the arrangement-complete negative certificate, off-the-shelf planner integration, attack suites, and the full sweep add roughly one to two more days. The negative certificate is the schedule-critical item (§8.4).

---

## 1. Purpose and design philosophy

### 1.1 Research context

Task-and-motion planning (TAMP) separates planning into a discrete **skeleton** (a symbolic action sequence with continuous parameters unbound) and **refinement** (sampling grasps/placements via "streams"). Learned accelerators such as PIGINet (Yang et al., 2023 — a feasibility classifier that reranks enumerated skeletons from scene image + plan tokens + goal literals) and LAZY (Khodeir et al., 2023 — a graph-attention per-step policy inside Levin tree search, with online success statistics and refinement-work sharing across skeletons) both inherit a representational assumption: that skeleton feasibility is predictable from per-object and pairwise features. The hypothesized research gap: domains where feasibility hinges on a **global, high-interaction-order, continuous geometric statistic** — here, whether a chosen *subset* of items jointly packs into a limited buffer — which is not decomposable into low-order terms, has no cheap certificate, and reveals infeasibility only late in refinement, making every wrong skeleton choice expensive.

DD2D is the fast, controlled 2D instrument for testing whether that gap **actually occurs on a naturalistic task distribution** — not for demonstrating it on a rigged one. It is a diagnostic tier: cheap enough to build in a few days, honest enough that its answer (gap or no gap, and at what buffer tightness) is meaningful evidence about whether a corresponding 3D flagship domain is worth building.

⟦R:M8⟧ **Stated prior.** The shape library (§4) is convex-majority (four of seven families — can/bowl/box/pillcase — are convex; dumbbell/shoe/horseshoe carry genuine concavities), and the packing literature (and our own earlier analysis) says convex-instance packability is largely captured by low-order summaries and greedy insertion with restarts. The *expected* sweep outcome is therefore that the direction is falsified at most λ, with any interesting regime confined to a band of tight slack and concavity-bearing candidate subsets. The instrument is designed to return either answer credibly; all results are stratified by measured slack ratio and by concavity presence (§11–12) so that if a regime exists, we can say precisely where, and if it does not, the negative is sharp.

### 1.2 The task

A household drawer, viewed top-down, holds 9–14 rigid items; one is the target. The robot uses a top-down parallel-jaw gripper: a grasp requires two finger footprints flush against opposite sides of an item, both collision-free. The target starts ungraspable — neighbors block every finger placement. Blockers may be staged onto a **counter-edge buffer**, a strip of open counter next to the drawer whose size is sampled from a plausible range. The decision each episode: **which subset of blockers to stage**, given that the staged set must jointly fit on the buffer *and be stageable and extractable by the actual gripper* (⟦R:M1,M2⟧ — the ground truth of §8 certifies executability, not bare geometric packing).

### 1.3 Measured, not installed — the load-bearing philosophy

This design descends from an earlier, heavily engineered version whose generator *enforced* every desired difficulty property (engineered decoy subsets that defeat area bounds, equal-area pieces to zero out size information, mutual-information caps between single-object features and labels, discard rules for marginal candidates, a one-way tray to manufacture irreversibility). That version was internally rigorous and externally worthless: its difficulty existed only because the generator installed it, so any method results said nothing about robotics. **DD2D inverts the role of the property list.** The generator produces naturalistic scenes; the desired properties are then *measured on the distribution* and reported as curves and strata:

| Property | Engineered approach (rejected) | DD2D (this spec) |
|---|---|---|
| No low-order certificate | decoys constructed to defeat area bounds; MI caps enforced at generation | Tier-0 model and heuristic-certificate accuracy **measured** vs. buffer tightness (§11) |
| Bimodal candidate feasibility | marginal candidates ⇒ instance discarded | marginal candidates **kept**, labeled, reported as a stratum (§8.5) |
| No degenerate shortcut | one-way slot; capacity engineered below evacuation | buffer size sampled from a natural range; shortcut usage **measured** (D6); slack-ratio ordering is a *required baseline* (§10.2 ⟦R:M5⟧) |
| Costly late failure | decoys designed so early placements succeed | failure-depth distribution **measured** (D2) |
| Skeleton diversity | exact-size-k candidate constraint | candidate count/size distributions **measured** |

The only generation-time filters are **decision-relevance** filters (§9.4): the target must actually be blocked, at least two distinct minimal clearing subsets must exist, and at least one candidate must be confidently feasible (a solvability certificate). These are the 2D analog of standard practice in the NAMO/rearrangement literature (filtering to scenes where obstacles actually obstruct), they are disclosed, their acceptance rates are reported, **and the distribution shift they induce is itself measured** (§9.5 ⟦R:m1⟧). Nothing about *labels* is filtered on beyond F3's existence requirement.

Consequence for interpretation: the central experiment (§11) sweeps buffer scale λ and plots headroom, attack accuracy, and certificate reliability as functions of λ *and of measured slack*. If a large planner-vs-oracle gap with weak shallow predictors appears at defensible tightness, the research direction is supported; if the gap only opens at implausible tightness, the direction is **falsified early and cheaply** — a success of the instrument, to be reported as such.

---

## 2. Conventions

- Units: centimeters, world frame, x–y plane, top-down view. Angles in radians internally, degrees in this document.
- All footprints are Shapely polygons (curved shapes polygonized at 24–32 vertices). "Inflate by r" = `poly.buffer(r, join_style='round')`; "deflate by r" = `poly.buffer(-r)` (⟦R:m6⟧ deflation of thin features can produce empty or multi-part geometry; the labeling code MUST assert the result is a non-empty single polygon and treat violations as a hard error — at current dimension ranges no library shape violates this at δ/2 = 0.5 cm, and a unit test pins that). Inflate *items*, never containers.
- ⟦R:m2⟧ **Two cost units, both instrumented everywhere.** (i) A **stream call** = one invocation of any sampler or test stream (§6.3) — the unit of planner-level accounting. (ii) An **elementary geometric evaluation (EGE)** = one polygon–polygon intersection/containment predicate evaluation — the unit that makes heterogeneous stream calls and the checker-in-loop baseline comparable (one `sample-buffer-pose` call costs ~m_p EGEs; one test stream costs 1 EGE; the nesting checker's EGEs are counted by the same counter). Wall-clock is reported secondarily with hardware stated. Any "equivalent stream calls" conversion is defined as EGEs ÷ (mean EGEs per stream call, measured on the same split) and labeled as such wherever used.
- Every stochastic component takes an explicit seed recorded in the instance JSON. Identical seeds ⇒ identical traces (a hard requirement; test it; pin Shapely/GEOS/NumPy versions).

## 3. Parameter table

Starred (*) items are the sweep/dial parameters of §11; the rest are fixed defaults.

| ID | Parameter | Default | Notes |
|---|---|---|---|
| P1 | Drawer interior W × D | W ~ U[35, 50], D ~ U[28, 40] cm | per instance |
| P2 | Wall band thickness | 1.5 cm | ring around interior; blocks fingers (§5.2) |
| P3 | Buffer strip L × d | L ~ U[25, 45], d ~ U[12, 20] cm, both × λ | adjacent to drawer, open (no walls) |
| P4* | Buffer scale λ | sweep {0.75, 0.9, 1.0, 1.1, 1.25} | the generator tightness dial; analysis also uses measured slack s (§11 ⟦R:M8⟧) |
| P5 | Item count N (incl. target) | U{9 … 14} | |
| P6 | Fill fraction f | Σ item areas / drawer area ∈ [0.35, 0.55] | sampled, then items drawn until reached |
| P7 | Finger width w_f | 1.5 cm (+0.5 clearance/side → 2.5 effective) | tangential extent |
| P8 | Finger thickness t_f | 1.0 cm (+0.5/side → 2.0 effective) | normal extent; note t_f,eff > δ — this is why labels certify *accessible* packings, not bare packings (§8.2 ⟦R:M1⟧) |
| P9 | Max aperture | 12 cm | min 0.5 cm |
| P10 | Grasp direction grid | 18 directions (10° steps over [0°, 180°)) | |
| P11 | Slide samples per direction | 5 | interior points along the contact-overlap interval (§5.3) |
| P12 | Label margin δ | 1.0 cm | §8.3 |
| P13* | Refinement budget B | 300 stream calls per skeleton (Tier 2); sensitivity cells at B ∈ {100, 1000} at λ* (⟦R:M4⟧); Tier-1 budget per §10.1 | |
| P14 | Per-step retry cap t | 10 before backjump | |
| P15 | Sampler strength m_p | 15 candidate poses per placement call | secondary dial |
| P16 | Checker rotation grids | 15° (positive search); Δθ_o = δ/(4·r_max(o)) (negative certificate) | §8.3–8.4 |
| P17 | Target placement prior | uniform over central 50% × 50% of interior | disclosed placement prior |
| P18 | Conceptual heights (h_item, h_wall, z_g, z_c) | 6, 12, 3, 15 cm | analytic only; never simulated (§5.1) |
| P19 | Labeling budget per candidate | 5 s wall-clock or 10⁵ checker EGEs | timeout ⇒ marginal with reason code (§8.4 ⟦R:M3⟧) |

## 4. Shape library — parametric household footprints

No engineered puzzle pieces. Items come from parametric families whose dimension ranges are anchored to common product sizes; each instance samples family + dimensions + small shape noise. Seven families — four convex, three with genuine (non-nub) concavities:

1. **Can** — circle, diameter 4–8 cm (small-medium).
2. **Bowl** — circle, diameter 8–12 cm (medium-large; capped near the aperture).
3. **Box** — rectangle 5–20 × 4–12 cm; 50% sharp-cornered, 50% rounded 0.3–1.0 cm (absorbs the old board/tray range; rejected if > 45% of drawer's short side²).
4. **Pillcase** — capsule (rectangle + semicircular ends), 10–18 × 2–4 cm.
5. **Dumbbell** — two identical end blocks (3–5 × 4–7 cm) joined by a thinner, longer bar (length 4–8, thickness 1.5–2.5 cm) → **concave waist**.
6. **Shoe** — an L of two similarly-sized rectangles (equal arm thickness 3–5 cm, arms 7–11 cm), one arm's long side flush with the other's short side → **concave inner corner**.
7. **Banana** — a C-shaped thick circular arc (outer radius 5–7 cm, thickness 2–3 cm, ~110–150° opening) → **concave C-opening**.

Sampling weights roughly uniform with boxes/cans slightly upweighted. Every sampled shape must admit ≥ 1 grasp in isolation (some direction with width ≤ aperture and a non-empty contact-overlap interval per §5.3) — resample otherwise; the concave families lose most grasp directions (as the old L-tool did) but keep ≥ 1. Two library splits by seed: **train** families/dimension draws and a **held-out** split with shifted dimension ranges (±15%) and one family swapped, for the generalization diagnostic (D4) and the holdout-generator audit (§10.4). Note what is deliberately *absent*: no equal-area constraint, no forced concavity, no complementarity engineering — the concave families are naturalistic household footprints, not difficulty injectors. Whatever complementarity structure exists in real household shapes is what the benchmark gets — that is the point. ⟦R:M8⟧ Each item records a `concave` flag (families 5–7: dumbbell/shoe/horseshoe); every candidate records whether S contains a concave item, and all label-dependent results are reported both pooled and stratified on this flag — *stratified in analysis, never steered in generation*.

## 5. World model

### 5.1 Height-stratified SE(2) × lift (the analytic third dimension)

DD2D is **not 2D physics**. It is the standard SE(2)×lift manipulation abstraction in which the *resting-pose* state space is 2D and the third dimension is handled analytically through a fixed height stratification (P18): items are prisms of uniform conceptual height h_item = 6 cm; drawer walls stand h_wall = 12 cm; grasping fingers descend to grasp depth z_g = 3 cm (so a finger occupies heights [3, ∞) for collision purposes); a carried item travels with its underside at carry height z_c = 15 cm. No z-coordinate is ever simulated or stored — the stratification exists solely to derive, once, which 2D collision query each manipulation phase requires:

| Phase | Height reasoning | 2D collision query performed |
|---|---|---|
| Grasp (fingers descend) | fingers [3,∞) overlap items [0,6] and walls [0,12] | finger rectangles vs. all other item footprints ∪ wall band |
| Lift / lower (vertical) | straight vertical motion at a fixed (x,y) | none beyond the grasp/place queries at the endpoints |
| Transfer (planar, carried) | item underside at 15 > wall top 12 > item top 6 | **none** — collision-free by the height argument |
| Place (item + fingers descend) | as grasp, at the destination | item footprint vs. destination-region contents ∪ container bounds; finger rectangles vs. destination-region contents ∪ wall band |

Consequence one: the motion stream is a constant-true test (still one accounted call, for uniform accounting) — this is a *derived* fact of the stratification, not an assumption, and it deliberately concentrates all continuous difficulty in grasps and placements so measured difficulty is attributable to the subset choice and its packing consequence. Consequence two, for rendering (§13): during transfer the carried item's footprint may overlap other items *in projection*, which is semantically fine (different heights) but visually reads as a collision bug; the renderer therefore MUST draw carried items in a distinct "elevated" style. Both facts belong to the world model, which is why they are specified here rather than discovered in the animation code.

### 5.2 Workspace

Drawer interior rectangle at the origin; wall band = the 1.5 cm ring around it (Shapely: outer rectangle minus interior). Buffer strip placed 6 cm to the drawer's right, sampled dimensions (P3 × λ). The buffer has no walls or lip: only already-staged items obstruct placements and finger approaches there (finger rectangles may overhang the strip's painted boundary — only item and wall-band collisions matter). There is no other placement surface — samplers never emit poses outside drawer/buffer.

### 5.3 Grasp model (uniform for polygons and circles)

A grasp of item o is g = (θ_i, s_j): θ_i from the 18-direction grid, s_j one of 5 slide positions. Construction: rotate the footprint by −θ; its axis-aligned x-extent gives two supporting lines; width = extent; admissible iff 0.5 ≤ width ≤ 12 cm. ⟦R:m6⟧ **Contact-overlap interval (precise definition):** let I_L, I_R ⊂ ℝ be the y-projections of the footprint's contact sets with the left and right supporting lines respectively (each contact set = footprint ∩ supporting line, possibly several segments); the contact-overlap interval is the y-interval hull of I_L ∩ I_R. If I_L ∩ I_R = ∅ (e.g., an L-tool at an angle where the two lines touch disjoint features), the direction is inadmissible. Slide positions s_j are drawn from the interior 80% of the contact-overlap interval. The two **finger rectangles** (2.5 × 2.0 cm effective, per P7–P8) sit flush against the supporting lines, centered at s_j; transform back to world. ⟦R:m7⟧ *Abstraction disclosed:* supporting-line contact does not model frictional force closure — a grasp on a slanted edge is treated as valid; DD2D's grasp validity is a kinematic-clearance abstraction, stated as such in any publication. Grasps are defined in the item frame, so the same g is reusable at any pose — but **all collision facts about a grasp are pose-dependent and certified only by the pose-explicit test streams of §6** (⟦R:M6⟧). Natural occlusion follows: an item hemmed in by neighbors or flush against the wall band loses grasp directions; a buried target loses all of them.

## 6. PDDLStream domain encoding

### 6.1 Certified facts, streams, and optimistic semantics

Continuous values (grasps, poses) enter the symbolic layer only as outputs of declared streams, each of which *certifies* facts about its outputs. This is the standard PDDLStream contract, and it is what makes skeletons well-defined before refinement: the symbolic planner plans over **optimistic objects** (placeholder outputs `#g1, #p1, …` assumed to satisfy exactly their certified facts), so *a skeleton is a symbolic plan that is valid under the optimistic closure of the stream declarations*. Refinement is the attempt to bind those placeholders with real stream evaluations; a skeleton "fails to refine" when some required certification cannot be produced within budget. Geometric validity is therefore never a bare precondition predicate — every geometric fact in a precondition is traceable to a stream that can certify it.

⟦R:M6⟧ **Static geometry as a pseudo-object.** The wall band is encoded as a static object `wall` with a permanent fact `(AtPose wall p_wall)` and no `(Item wall)` fact (so it can never be picked or sampled). All static-geometry clearance is then covered *uniformly* by the same CFree machinery as item–item clearance — no pose-independent prefilters, no special-cased wall checks, no stale facts.

```
;; stream.pddl  (v3: pose-explicit grasp-clearance; no pose-dependent claims in sample-grasp)
(:stream sample-grasp
  :inputs (?o) :domain (Item ?o)
  :outputs (?g) :certified (Grasp ?o ?g))
      ; emits admissible (θ, s) cells only (aperture + non-empty contact-overlap
      ; interval, §5.3). NO collision facts are certified here: grasp clearance
      ; is pose-dependent and belongs to test-cfree-grasp below.   ⟦R:M6⟧
(:stream sample-buffer-pose
  :inputs (?o) :domain (Item ?o)
  :outputs (?p) :certified (BufferPose ?o ?p))
      ; compaction-biased (§6.3); emits only poses with footprint ⊆ buffer
(:stream sample-drawer-pose
  :inputs (?o) :domain (Item ?o)
  :outputs (?p) :certified (DrawerPose ?o ?p))
      ; footprint ⊆ drawer interior (wall-band clearance of the FOOTPRINT is
      ; guaranteed by containment; FINGER clearance vs. the wall is covered by
      ; test-cfree-grasp against the `wall` pseudo-object)
(:stream test-cfree-pose-pose
  :inputs (?o1 ?p1 ?o2 ?p2) :domain (and (PoseOf ?o1 ?p1) (PoseOf ?o2 ?p2))
  :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2))
      ; footprints at these poses do not overlap
(:stream test-cfree-grasp
  :inputs (?o ?g ?p ?o2 ?p2)
  :domain (and (Grasp ?o ?g) (PoseOf ?o ?p) (PoseOf ?o2 ?p2))
  :certified (CFreeGrasp ?o ?g ?p ?o2 ?p2))
      ; fingers for grasping/releasing o AT POSE p clear o2's footprint at p2.
      ; ONE pose-explicit test serves both pick (p = current pose) and place
      ; (p = destination pose); v2's pose-implicit CFreeGraspPose is removed
      ; because with place-drawer legal an item can be re-picked at a new pose,
      ; making any pose-implicit fact stale.                        ⟦R:M6⟧
```

(`PoseOf` abbreviates "is a pose value bound to this object" — initial poses (including `p_wall`), BufferPose, or DrawerPose outputs.)

### 6.2 Domain

```
;; fluents: (AtPose ?o ?p) (Holding ?o ?g) (HandEmpty) (InDrawer ?o) (OnBuffer ?o) (Extracted ?o)
;; static: (Item ?o) (Target ?o) + all certified facts above; `wall` has AtPose but not Item

(:action pick
  :parameters (?o ?p ?g)
  :precondition (and (HandEmpty) (AtPose ?o ?p) (InDrawer ?o) (Item ?o) (Grasp ?o ?g)
                     (not (UnsafeGraspAt ?o ?g ?p)))
  :effect (and (Holding ?o ?g) (not (AtPose ?o ?p)) (not (InDrawer ?o)) (not (HandEmpty))))

(:action place-buffer
  :parameters (?o ?g ?p)
  :precondition (and (Holding ?o ?g) (BufferPose ?o ?p) (not (UnsafePlace ?o ?g ?p)))
  :effect (and (AtPose ?o ?p) (OnBuffer ?o) (HandEmpty) (not (Holding ?o ?g))))

(:action place-drawer          ; legal — see §6.4
  :parameters (?o ?g ?p)
  :precondition (and (Holding ?o ?g) (DrawerPose ?o ?p) (not (UnsafePlace ?o ?g ?p)))
  :effect (and (AtPose ?o ?p) (InDrawer ?o) (HandEmpty) (not (Holding ?o ?g))))

(:action retrieve
  :parameters (?o ?p ?g)
  :precondition (and (HandEmpty) (Target ?o) (AtPose ?o ?p) (InDrawer ?o) (Grasp ?o ?g)
                     (not (UnsafeGraspAt ?o ?g ?p)))
  :effect (and (Extracted ?o) (not (AtPose ?o ?p)) (not (InDrawer ?o))))

(:derived (UnsafeGraspAt ?o ?g ?p)              ; pose-explicit ⟦R:M6⟧
  (exists (?o2 ?p2) (and (AtPose ?o2 ?p2) (not (= ?o ?o2))
                         (not (CFreeGrasp ?o ?g ?p ?o2 ?p2)))))
(:derived (UnsafePlace ?o ?g ?p)
  (exists (?o2 ?p2) (and (AtPose ?o2 ?p2) (not (= ?o ?o2))
                         (or (not (CFreePosePose ?o ?p ?o2 ?p2))
                             (not (CFreeGrasp ?o ?g ?p ?o2 ?p2))))))
      ; the ?o2 quantification ranges over the `wall` pseudo-object too, so
      ; finger-vs-wall clearance at both pick and place needs no special case.

;; goal: (exists (?o) (and (Target ?o) (Extracted ?o)))
```

Blocking is thus never a static literal: it *emerges* from the Unsafe/CFree machinery evaluated against current poses. A precomputed `blocks(o, target)` initial-state encoding is forbidden — it would hand the clearing structure to the symbolic layer as a pairwise set-cover and collapse candidate diversity; with this encoding, geometry stays in the streams and the symbolic layer stays geometry-blind. A pleasant structural consequence: the shortest optimistic plan is literally `retrieve(target, p0, #g)` — "just grab it" — which fails when the UnsafeGraspAt tests cannot be certified against the neighbors, after which the planner grows longer plans. Honest uninformed enumeration therefore begins with over-optimistic short plans and expands, which is the enumeration behavior the research premise assumes, obtained for free rather than scripted.

### 6.3 Stream implementations

`sample-grasp`: seeded draw from o's admissible (θ, s) cells (§5.3; no collision filtering ⟦R:M6⟧). `sample-drawer-pose`: uniform over drawer free space (kept weak; it serves the rarely-useful place-drawer). `test-*`: exact Shapely intersection checks (1 EGE each). `sample-buffer-pose` is the refiner's key component and is **compaction-biased** — a uniform sampler wastes buffer space on early placements and fails even on feasible subsets at tight λ:

```
function sample_buffer_pose(o, staged, rng, m_p=15):
    θ ~ 15° grid + jitter U(±7.5°)
    candidates = []
    repeat m_p times:
        with prob 0.7:   # contact proposal
            pose with o(θ) touching a staged item or a buffer edge, offset outward
            by ε = 0.15 cm (sampled on the contact boundary of the free region)
        with prob 0.3:   # slide proposal
            uniform free pose, translated toward nearest contact along a random
            direction, backed off ε
        keep if collision-free
    # ⟦R:m6⟧ scalar score (v2's lexicographic-plus-noise was ill-defined):
    score(pose) = pose.x + 0.01 · pose.y + Gumbel(0, β = 0.3)      # cm units
    return argmin score over candidates, or FAIL if none collision-free
```

The sampler is deliberately *incomplete and cheap*: on infeasible subsets it can only fail-to-find, which consumes budget — that expensive failure is the cost structure under study, not a bug. Its strength m_p is a secondary dial. All samplers/tests are instrumented for stream-call **and EGE** accounting (⟦R:m2⟧) and consume (instance stream seed + per-skeleton nonce) for reproducibility.

### 6.4 `place-drawer` stays legal

In a naturalistic drawer, repositioning items into holes vacated by removed ones can occasionally help. Solvers may use it; how often generous-budget solutions actually do is diagnostic D6 — if the answer is "often," that is a finding about the distribution (and the controlled skeleton space of §10.2 gets revisited), not a reason to ban the action.

## 7. Geometric candidate enumeration (dataset infrastructure — not a planner)

The labeling harness needs the set of clearing candidates. It computes them **geometrically**, using information the symbolic layer deliberately lacks; it is therefore disclosed infrastructure, never presented as a baseline planner (§10 keeps the roles separate):

```
function enumerate_candidates(scene):
    cells = all (θ_i, s_j) grasp cells of the target
    for each cell c: blockers(c) = { o ≠ target : footprint(o) ∩ finger_rects(c) ≠ ∅ }
                     (cells whose rectangles hit the wall band are discarded)
    minimal = minimal sets under ⊆ among { blockers(c) }
    supersets = each minimal set ∪ one adjacent item (adjacency = footprint
                distance < 2 cm), added in seeded random order until ≤ 40 candidates
    for each candidate S, two validity re-checks (drop failures, report drop
    rate BY CAUSE):
      (a) clearing re-check: with S geometrically removed, the target has ≥ 1
          collision-free grasp (the blocker-set computation is first-order
          optimistic; this closes it)
      (b) extraction-order re-check ⟦R:M2⟧: ∃ an ordering of S such that each
          member, at its turn, has ≥ 1 grasp whose fingers clear all items not
          yet removed and the wall band. Exact search over orderings with
          memoization on the removed-subset lattice (≤ 2^6 states; trivial).
          Candidates with no extraction order are RETAINED but pre-labeled
          infeasible(reason=extraction) — planners face them, so the dataset
          keeps them; they are excluded from packing-attack strata (§10.4).
    published order: ascending |S|, ties by seeded random permutation
      (⟦R:M5⟧ note: ascending |S| is itself a weak packing heuristic; Tier-2
       therefore also runs random and slack-ratio orderings, §10.2)
    return candidates       # candidate ≡ staging skeleton: pick/place-buffer per member, retrieve
```

Candidate sizes vary naturally (typically 2–6); size may correlate with feasibility, and that information is available to attacks and methods alike — part of the distribution, measured rather than zeroed out.

## 8. Ground-truth labeling

### 8.1 Why forward-generate-then-label is sound here

Backward ("planted") generation exists to solve the label-and-solvability cost problem in 3D. In 2D at this scale, exact subset checks are affordable, so every candidate of every instance is labeled at generation time under an explicit compute budget (P19). This preserves planting's two benefits — a per-instance **solvability certificate** (filter F3) and a free **oracle bracket** (a planner that reads the labels) — without planting's leakage risks, and it avoids the bias of rejection-sampling through a search heuristic: filtering conditions only on exact label structure, never on what a solver finds easy.

### 8.2 What the labels certify: accessible packings ⟦R:M1⟧

v2 labeled bare packings; that is the wrong question, because the effective finger thickness (2.0 cm, P8) exceeds the label margin (δ = 1.0 cm): a subset can pack with δ-separations that no finger can enter, so a "feasible" label would not imply refinability, and the mismatch concentrates at tight λ — exactly where the headroom claim lives, where it would masquerade as sampler weakness. v3 labels therefore certify **accessible packings**:

> **accessible packing of S** ≔ a packing {pose(o)}₍o∈S₎ in the buffer **plus** an insertion order o₍₁₎,…,o₍|S|₎ **plus**, for each o₍i₎, a grasp g₍i₎ whose finger rectangles at pose(o₍i₎) are disjoint from the footprints of o₍₁₎,…,o₍i₋₁₎ at their poses (fingers may overhang the buffer's painted boundary; only items and the wall band collide, §5.2).

Two soundness facts that keep the machinery simple: (i) *no packing ⇒ no accessible packing*, so the packing-only negative certificate of §8.4 remains sound for the new label; (ii) the positive certificate is constructive — the checker records the witness order and grasps, so a feasible label comes with everything the spot-audit needs. The **pack-but-inaccessible stratum** (packs at margin, but no insertion order/grasp assignment found) is a reported quantity per λ (D9) — it is a genuine, interesting property of tight buffers, now measured instead of silently corrupting refiner-adequacy numbers.

The overall candidate label composes drawer-side and buffer-side executability:

> **feasible(S)** ⇔ extraction order exists (§7b) ∧ accessible δ-packing certificate found (§8.3)
> **infeasible(S)** ⇔ no extraction order ∨ packing-nonexistence certificate found (§8.4)
> **marginal(S)** ⇔ neither, with a recorded `reason ∈ {geometric, budget, inaccessible}` ⟦R:M3⟧

### 8.3 Positive certificate (margin-gap rule, sound for continuous poses)

**Packing search:** NFP-based depth-first nesting on the **δ/2-inflated** shapes, rotation grid 15°, shapes tried by descending area with 2 random restarts, candidate positions per §8.4's arrangement rule (using it on the positive side too costs nothing and reuses code). A found witness implies a real packing of the true shapes with ≥ δ pairwise separation and ≥ δ/2 boundary clearance — sound by inflation.

**Accessibility search:** given a witness packing, search insertion orders (≤ |S|! with removed-subset memoization); for each prefix, for the next item, scan its admissible grasp cells for one whose fingers clear the already-placed prefix at witness poses (exact intersection tests). If no order works, try the next packing witness (restart); after the restart budget, the candidate is *not* labeled feasible — it falls to §8.4's negative search, and if that also fails, marginal(reason=inaccessible).

The δ margin plus accessibility certificate together make staging order irrelevant *by construction* for feasible-labeled candidates; the v2 spot-audit (3 random refiner orders × 2 seeds on a 5% sample) is retained as a validation test of the whole chain, no longer as the only line of defense.

### 8.4 Negative certificate (arrangement-complete + Lipschitz rotation grid) ⟦R:M3⟧

**infeasible-by-packing(S)** ⇔ exhaustive search finds no packing of the **δ/2-deflated** shapes over the per-shape rotation grid Δθ_o = δ/(4·r_max(o)) (r_max = max centroid-to-boundary distance), with candidate positions taken from the **arrangement vertex set**:

> At each fixed rotation assignment, when placing item i given already-placed items P: the free region for i's reference point is the inner-fit polygon (IFP) minus ⋃₍j∈P₎ NFP(i, j). Candidate positions = all vertices of the *arrangement* of these boundary curves — i.e., IFP vertices, NFP vertices, **and every pairwise intersection point between two NFP boundaries or an NFP boundary and the IFP boundary** — filtered to points on the free-region boundary.

*Why v2's rule was unsound:* v2 searched only individual NFP boundary vertices and edge midpoints. The classical completeness result for translational nesting is that if any packing exists, one exists with every item in contact, at a vertex of the free-space arrangement — which includes NFP–NFP intersection points that are vertices of *neither* curve alone. Toy failure: a piece that fits only in the pocket formed jointly by two placed pieces and the container wall sits exactly at such an intersection; v2's DFS reports "no packing," producing a false infeasible label that contaminates every downstream number. The arrangement rule restores completeness at fixed rotations.

*Rotation-grid lemma (state it, unit-test it):* Suppose a continuous packing of the true shapes exists with pairwise clearance ≥ δ/2 and boundary clearance ≥ δ/2. Snap each shape's rotation to its nearest grid angle about its centroid: the angular change is ≤ Δθ_o/2 = δ/(8·r_max(o)), so every boundary point moves ≤ r_max(o)·δ/(8·r_max(o)) = δ/8. Pairwise clearance after snapping is ≥ δ/2 − 2·(δ/8) = δ/4 > 0 and boundary clearance ≥ δ/2 − δ/8 = 3δ/8 > 0; a fortiori the δ/2-deflated shapes (subsets of the true shapes) admit a non-overlapping contained placement at grid rotations, which the arrangement-complete translational search will find. Contrapositive: exhaustive failure certifies no continuous δ/2-clearance packing of the true shapes exists. (The constants carry ~2× slack; a unit test constructs a near-threshold instance and checks both directions numerically.)

*Label semantics, stated plainly:* feasible = packable-and-stageable with ≥ δ separation; infeasible = not packable even at δ/2 clearance (or not extractable at all); marginal = the gap between. A subset packable only at clearances below δ/2 is deliberately *not* called feasible — the refiner's ε = 0.15 cm contact offsets could not realize it anyway.

**Compute policy (v2's "milliseconds to ~1 s" was not credible for 5–6 item negatives at Δθ ≈ 1°):** run cheap-to-expensive per candidate — (i) extraction-order check; (ii) H1 area bound as a *sound* infeasibility shortcut (Σ deflated areas > buffer area ⇒ infeasible; sound pruning inside the DFS too: remaining deflated area > remaining free area ⇒ prune — area bounds are sound prunes; greedy heuristics are NOT and never gate a label); (iii) positive search + accessibility; (iv) exhaustive negative search under budget P19 (5 s / 10⁵ EGEs). **Timeout ⇒ marginal(reason=budget), never ⇒ infeasible.** Labeling-time distribution and timeout rates are reported per λ; if budget-marginals exceed ~10% of candidates at any λ, raise P19 and disclose.

**Day-1 fallback** (if the arrangement machinery slips the schedule): anytime DFS with contact-point candidates and a generous node cap, labeling "found / not-found-at-cap" with not-found flagged *provisional*; upgrade before running attacks or reporting any label-dependent numbers — provisional negatives are not sound.

### 8.5 Marginal candidates

**Kept, not discarded.** Marginal candidates are part of the naturalistic distribution: planners face them; the marginal fraction per λ (with reason breakdown) is a reported stratum; learned-attack and oracle evaluations are reported on the confidently-labeled subset (stated wherever those numbers appear).

## 9. Instance generator

### 9.1 Scene synthesis (forward)

```
function generate(seed, λ):
    rng = RNG(seed)
    sample drawer dims (P1), buffer dims (P3 × λ), fill fraction f (P6)
    place target: family+dims sampled; pose uniform over central 50%×50% (P17), any rotation
    while coverage < f and items < 14:
        sample item; place by settled-clutter procedure:
            uniform proposal in free space → translate toward nearest contact
            (wall band or existing item) along a random direction, back off 0.2 cm
            → small rotation jitter; reject on overlap; 30 tries then skip item
    return scene
```

The settle-toward-contact step makes scenes read as real drawers (items leaning into each other and the walls) rather than Poisson confetti; it is a placement prior, disclosed, identical for target-adjacent and distant items — nothing steers blockers toward the target beyond density itself.

### 9.2 Labeling pass

Enumerate candidates (§7, including both re-checks), label every candidate per §8 under budget P19, record everything: labels with reasons, witness poses **and witness insertion orders/grasps** for feasible candidates, labeling wall-clock and EGEs per candidate.

### 9.3 Ordering of checks ⟦R:m6 — v2 skipped §9.3⟧

Filters (§9.4) are evaluated only after the *full* labeling pass, because the extraction re-check (§7b) can shrink the minimal-subset structure F2 depends on, and F3 reads final labels. Never filter on intermediate label states.

### 9.4 Decision-relevance filters (the only filters)

- **F1 — target blocked:** no valid (collision-free) grasp of the target exists in the initial scene.
- **F2 — real choice:** ≥ 2 distinct minimal clearing subsets survive the §7 re-checks.
- **F3 — solvability certificate:** ≥ 1 candidate labeled confidently feasible (§8.2 composite label).

Instances failing any filter are discarded; **acceptance rates per filter, per λ, are reported** (expected order of magnitude at defaults: F1 passes 30–60% given the central target prior and fill range; F2∧F3 conditional pass 40–80% — measurements, not targets). No filter conditions on label *patterns* beyond F3's existence requirement, on attack performance, or on planner behavior.

### 9.5 Filter-shift audit ⟦R:m1⟧

The filters select on difficulty structure, and F3 touches labels; disclosure alone does not quantify the induced shift. On a 500-instance pre-filter sample per λ: compute the Tier-0 feature vector (§10.4) of every enumerated candidate and scene-level summaries (fill fraction, item count, mean local density at the target); report accepted-vs-rejected distribution comparisons (per-feature standardized mean differences + a 2-sample classifier AUC "distinguishability" score). This audit is descriptive — it bounds how much of any measured structure could be filter-induced rather than natural, and it ships with the dataset.

### 9.6 Instance JSON (schema sketch)

```json
{"instance_id":"dd2d-000042","seeds":{"gen":42,"stream_base":420},
 "lambda":1.0,"drawer":{"w":41.2,"d":33.5},"buffer":{"x0":48.7,"l":31.0,"d":16.4},
 "items":[{"id":"o0","family":"can","params":{"dia":7.5},"concave":false,
           "polygon":[[..]],"pose":{"x":..,"y":..,"th":..},"is_target":false},...],
 "target_id":"o6",
 "candidates":[{"subset":["o1","o3"],"label":"feasible",
                "witness":{"poses":[..],"order":["o1","o3"],"grasps":[..]},
                "slack_ratio":0.71,"contains_concave":false},
               {"subset":["o1","o4"],"label":"infeasible","reason":"packing"},
               {"subset":["o2","o5"],"label":"infeasible","reason":"extraction"},
               {"subset":["o2","o3","o5"],"label":"marginal","reason":"budget"}],
 "filter_report":{"F1":true,"F2":true,"F3":true},
 "labeling_cost":{"wall_s":3.2,"eges":48211},
 "published_order":[0,2,1,...]}
```

Splits: train / val / test / held-out-shape (library holdout, §4) / holdout-generator (§10.4), **split at the instance level** — candidates from one instance never straddle a split (⟦R:m4⟧). PDDLStream problem files (domain.pddl, stream.pddl, per-instance init including the `wall` pseudo-object) are emitted alongside the JSON so Tier-1 planners consume instances directly.

## 10. Planner evaluation: a two-tier protocol

The two tiers answer different questions and must not be conflated. **Tier 1** asks: what does this distribution cost a real, unmodified TAMP planner? **Tier 2** asks: holding the candidate space fixed and fully labeled, how much of that cost is attributable to skeleton *selection*, and how well do methods and shallow predictors close it? Tier-1 headroom numbers will be larger and noisier than Tier-2 numbers (real planners also spend budget discovering *which subsets clear*, and on optimistic short plans); both are legitimate. Reporting leads with Tier 2 for the mechanism claim and gives Tier 1 as the practical-planner cost.

### 10.1 Tier 1 — off-the-shelf PDDLStream baselines

Run unmodified PDDLStream algorithms (incremental, focused/binding, adaptive; Fast Downward backend) on the shipped domain/stream files, at a generous total budget (Gate-1 measurement: solve rate should approach 100% given F3 and refiner adequacy — the residual is itself reported) and at a **practical budget defined as: the per-λ median Tier-2 published-order total stream-call spend, computed on the val split and frozen before test runs** (⟦R:m3⟧). Stream wrappers instrument call and EGE counts identically to Tier 2. Report, per λ: solve rate, stream calls / EGEs / wall-clock per solve, and a **failure decomposition** — refinement effort classified post-hoc (using labels and geometry) into: optimistic short plans (e.g., bare `retrieve`), plans whose staged subset does not clear, plans whose subset does not admit an extraction order, plans whose subset clears but does not pack (or is inaccessible), and productive `place-drawer` plans (D6). This decomposition is what makes the packing-selection difficulty separable from clearing discovery in an honest off-the-shelf run.

### 10.2 Tier 2 — controlled comparisons on the shared candidate set

All Tier-2 planners see the same disclosed, geometry-informed candidate set (§7) and share the refiner below; they differ only in candidate *ordering/selection policy*, which isolates the variable under study and is the fair arena for learned methods (rerankers need an enumerated set; per-step policies get the same set as a selection problem).

**Refiner** (budget B per skeleton): sequential binding with backjumping over a candidate S. ⟦R:M7⟧ **Within-candidate ordering policy (previously unspecified):** members are attempted in a greedy-graspable order — at each step, try members of the remaining set in seeded random order and bind a pick for the first member with a certifiable grasp (sample-grasp + CFreeGrasp tests at its current pose); then bind its place-buffer (sample-buffer-pose + CFreePosePose + CFreeGrasp at the destination); on t = 10 consecutive failures at a step, backjump: undo the previous placement and re-sample it, and permit member-order backtracking (re-choosing which member goes next); all attempts count against B; the first step retries until budget. Finish with retrieve. This policy is part of the shared infrastructure — every Tier-2 variant and any learned method uses it unchanged. The expected signature on infeasible subsets — early placements succeed, the last fails, backjumps thrash, budget exhausts — is *measured* by D2, not assumed.

**Variants (all required):**
1. **published-order** — the §7 order (ascending |S|); ⟦R:M5⟧ renamed from v2's "uninformed" because ascending size is itself a weak packing heuristic;
2. **random-order** — uniformly permuted candidates (the genuinely uninformed floor);
3. **slack-order** ⟦R:M5⟧ — candidates ascending by slack ratio Σ area(S) / buffer area, ties by |S|: the strongest zero-training cheap ordering a practitioner would write in five minutes. **Headroom claims are made against the best of orderings 1–3** — if slack-order alone closes the gap, that is the answer and it is reported as such;
4. **oracle** — refines the first confidently-feasible candidate read from labels (upper bracket; uses the shared refiner and *not* the witness poses/order — the oracle knows *which*, not *how*);
5. **checker-in-loop** — calls the anytime nesting checker (capped, e.g., 0.2 s or 10⁴ EGEs per candidate) to skip rejected candidates; accounted in wall-clock, EGEs, and equivalent stream calls (§2); expected strong in 2D and pre-registered as such — the learned-method question is amortization cost and scaling;
6. **constant policy** — best fixed rule fit on train (e.g., nearest-k blockers), measuring degenerate-shortcut availability;
7. **retrieval baseline** — nearest training instance by shape-multiset + buffer descriptor, reuse its feasible subset mapped by shape similarity (memorization headroom, D3).

### 10.3 Coverage audit (protects the oracle bracket)

On a sample of instances, run a Tier-1 planner at generous budget and map every solution found to the labeled candidate set: exact candidate / superset of a candidate / uses place-drawer / other. Report the distribution. If a non-trivial mass lands in "other," the labeled space under-covers what real planners do, the oracle bracket is an oracle over the wrong space, and the enumerator must be extended before Tier-2 claims are made.

### 10.4 Attack suites (measurements, run per λ)

**Heuristic certificates:** H1 summed-area bound (Σ area(S) ≤ buffer area; note H1 is one-directional — sound for infeasibility only); H2 one-shot greedy bottom-left insertion (30° grid, descending area); H3 = H2 with 3 restarts; H4 bounding-box shelf bound. Balanced accuracy on confidently-labeled candidates (base rates are skewed; raw accuracy misleads).

**Tier-0 learned attack:** logistic regression + gradient-boosted trees over low-order features — per-object (area, perimeter, circularity, max caliper width), pairwise within S (pair NFP-area complementarity), aggregates (|S|, Σ area, max single dimension, buffer slack ratio). Fixed training budget, standard hyperparameter search, no representation learning, instance-level splits (⟦R:m4⟧). This operationalizes "no low-order sufficient statistic" — as a measured curve, not a construction constraint.

⟦R:M2⟧ **Two attack targets, both reported:** (i) the overall composite label (what a planner-facing predictor must predict); (ii) the packing-only label restricted to extraction-feasible candidates. The distinction matters because extraction infeasibility is plausibly predictable from low-order local-blocking features; conflating the two would let extraction structure inflate (or deflate) the "no low-order predictor of *packing*" measurement, which is the claim under test.

**Mid-tier model (optional, day 3+):** a small set-transformer over per-shape encodings of S + buffer descriptor; its position between Tier-0 and oracle informs the representational-vs-integration framing downstream.

**Filter-artifact audit:** (i) the Tier-0 attack *is* the leakage measurement — any shallow correlate of labels, whether from nature or from filtering, shows up there; (ii) the §9.5 filter-shift audit quantifies the induced covariate shift directly; (iii) the **holdout generator** (shifted dimension ranges, swapped family, different fill band) provides a transfer split on which only method *rankings*, not absolute gaps, are claimed.

## 11. The buffer-slack sweep (the central experiment)

For each λ ∈ {0.75, 0.9, 1.0, 1.1, 1.25}: generate 300 accepted instances (logging acceptance rates and the §9.5 audit), run both tiers and the attack suites, and analyze against **two abscissae**: the generator dial λ, and ⟦R:M8⟧ the **measured slack ratio** s(instance) = min over confidently-feasible candidates of Σ area(S)/buffer area (with the per-candidate slack_ratio recorded in the JSON) — λ conflates buffer size with demand; s is the physically interpretable variable, and the headline plots bin on s with λ shown as marker color. Quantities:

- **(a) Refiner adequacy:** refinement success rate on confidently-feasible candidates (5 stream seeds). If it sags below ~0.85–0.90 at tight λ, raise m_p before interpreting anything else — with accessible-packing labels (§8.2), residual failures are attributable to the sampler, which is exactly what this dial isolates.
- **(b) Shallow predictability:** best heuristic and best Tier-0 balanced accuracy, on both attack targets of §10.4, on confidently-labeled candidates.
- **(c) Selection headroom, budget-independent form ⟦R:M4⟧:** the primary quantity is **excess failed refinements per solve** E = E[rank of first feasible candidate − 1] under the *best cheap ordering* (best of published/random/slack), with its full distribution (D1). The call-ratio "headroom" is then reported as the cost translation — uninformed-vs-oracle calls ≈ 1 + E·B/c_oracle, which is *affine in the experimenter-chosen B*; to make that dependence visible rather than exploitable, headroom is reported at B ∈ {100, 300, 1000} at λ*, alongside Tier-1 practical cost with its failure decomposition.
- **(d) Strata:** marginal-candidate fraction (with reason breakdown, incl. budget timeouts); pack-but-inaccessible fraction (D9); candidate count/size distributions; concavity-stratified versions of (b)–(c); D2 failure-depth profile; D6 place-drawer prevalence.

**Statistical protocol ⟦R:m5⟧:** all per-λ quantities carry 95% bootstrap CIs over instances; method comparisons are paired on shared instances (same candidate sets, same stream seeds) with Wilcoxon signed-rank p-values reported; refiner-adequacy uses per-candidate seed replicates. Go/no-go criteria are evaluated on point estimates with CIs displayed; a criterion whose CI straddles its threshold triggers a pre-registered N-extension (to 600 instances at the contested λ) rather than a judgment call.

**Interpretation (go/no-go, pre-registered, budget-robust form):** the research direction is supported at λ* iff, at λ*: (a) ≥ 0.85; (b) ≤ ~0.65 **on the packing-only attack target**; and (c′) excess failed refinements per solve under the *best cheap ordering* ≥ 2 (equivalently ≥ 3 refinement attempts per solve) — with λ* inside the plausible counter-edge band (roughly λ ≥ 0.9; a gap that exists only at λ = 0.75 with a ~9 cm-deep strip is an artifact of implausible tightness and must be reported as a negative result for the direction). Criterion (c′) replaces v2's "headroom ≥ 5×," which was purchasable by raising B. Runtime: generation ≈ 0.1–2 s per instance; **labeling is budgeted, not assumed fast** — negative certificates on 5–6-item subsets at Δθ ≈ 1° dominate, bounded by P19 per candidate (worst case ≈ 40 × 5 s ≈ 3.5 min per instance; expected far less since positive searches and area bounds settle most candidates); the Tier-2 sweep otherwise fits in well under an hour on a laptop; Tier-1 runs dominate wall-clock and can be sampled (e.g., 100 of 300 instances per λ) with the sampling disclosed.

## 12. Metrics and diagnostics (reporting spec)

Per method, per tier, per λ (and binned s) and split: solve rate, stream calls and EGEs per solve (mean/median/P90), failed refinements per solve, wall-clock (hardware stated), bracket table (Tier-1 practical / Tier-2 random / published / slack / learned method / checker-in-loop / oracle). Diagnostics: **D1** rank of first feasible candidate under each ordering (distribution, not just mean); **D2** per-rejection deepest step and call cost; **D3** retrieval-baseline solve rate and entropy of the feasible-set identity given Tier-0 features; **D4** held-out-shape and holdout-generator transfer (rankings only on the latter); **D5** per-heuristic balanced accuracy / TPR / FPR on both attack targets; **D6** productive `place-drawer` frequency in generous-budget Tier-1 solutions; **D7** Tier-1 failure decomposition (§10.1); **D8** coverage-audit distribution (§10.3); **D9** ⟦R:M1⟧ pack-but-inaccessible fraction per λ; **D10** ⟦R:m1⟧ filter-shift audit summary (§9.5).

## 13. Rendering and video evidence

Static top-down Matplotlib rendering: drawer walls, gray items, highlighted target, red translucent finger rectangles on blocked grasps, buffer with dashed witness/ghost outlines.

Animation (`FuncAnimation` → ffmpeg mp4, 20 fps) with the **elevated-carry convention** required by §5.1: a carried item is drawn as a dashed outline with no fill (optionally light hatching), with a small offset drop-shadow polygon at its ground projection, and a "carrying o3" overlay tag; resting items keep solid fills; finger rectangles are drawn only during grasp and place phases. This convention is load-bearing, not cosmetic — in projection a carried item may legitimately overlap resting items, and without the style switch every transfer reads as a collision bug to a viewer or reviewer. Phase tags in the per-frame JSONL event log drive the style switch. Failed placement samples flash as red ghosts at rejected poses; backjumps show the prior item lifting (style switch) and re-placing; overlays show candidate index, subset members highlighted in-scene, a running stream-call counter, and a verdict banner. Deliverables per demo instance: `success_<id>.mp4` (oracle run end-to-end), `failure_<id>.mp4` (one infeasible candidate: early placements succeed, the last item's red ghosts accumulate, budget exhausts), `session_<id>.mp4` (published-order-planner montage). Every video pairs with its event log for exact reproduction.

## 14. Software plan

### 14.1 Modules (~1,700–2,100 LOC total)

```
dd2d/
  shapes.py       # parametric families, polygonization, concave flag, splits  (~160)
  geometry.py     # inflate/deflate (+ non-emptiness asserts), convex decomp,
                  # NFP, arrangement vertices (NFP–NFP/IFP intersections)      (~260)
  grasps.py       # supporting-line grasp model, contact-overlap interval,
                  # finger rects, validity                                     (~140)
  nesting.py      # positive (inflated) + negative (deflated, arrangement-
                  # complete) search, accessibility search, budget policy,
                  # sound area pruning, anytime + day-1 fallback modes         (~330)
  scene.py        # drawer/buffer, settled-clutter synthesis                   (~150)
  enumerate.py    # candidate enumeration + clearing & extraction re-checks    (~150)
  label.py        # composite label rule, reasons, filters F1–F3, filter-shift
                  # audit, JSON I/O                                            (~170)
  streams.py      # samplers/tests, call + EGE accounting, seeding             (~170)
  refine.py       # backjumping refiner incl. member-ordering policy (§10.2)   (~120)
  tier2.py        # random / published / slack / oracle / checker-in-loop /
                  # constant / retrieval                                       (~170)
  tier1.py        # PDDLStream adapter: problem emission (incl. wall pseudo-
                  # object), instrumented wrappers, runners, failure decomp    (~190)
  domain/         # domain.pddl, stream.pddl (checked in, versioned)
  attacks.py      # heuristics + Tier-0 models, two attack targets             (~160)
  sweep.py        # λ sweep runner, slack-binned analysis, bootstrap CIs       (~140)
  render.py       # static rendering + elevated-carry animation                (~170)
```

Dependencies beyond the scientific stack: the `pddlstream` library and a Fast Downward build (pddlstream's bundled build script; budget ~30 minutes of setup and expect minor integration friction — this is why Tier 1 is scheduled on day 2–3).

### 14.2 Milestone plan (with an AI coding assistant)

**Day 1 (Tier-2 pipeline end-to-end):** Hours 1–2: `shapes` + `geometry` (unit tests: overlap, inflate-items-not-containers, deflation non-emptiness on every library shape at δ/2). Hours 2–4: `grasps` (unit test: L-tool direction with disjoint contact sets is inadmissible) + `scene` + static rendering; eyeball a dozen scenes for naturalness; `enumerate` with both re-checks (unit test: a hand-built scene where a candidate's member is itself buried ⇒ infeasible(extraction)). Hours 4–6: `nesting` (positive + accessibility; negative in fallback mode today) + `label` + filters; log acceptance rates. Hours 6–8: `streams` + `refine` + random/published/slack/oracle; first D1/headroom numbers at λ = 1.0 on ~50 instances.
**Day 2:** arrangement-complete negative certificate (unit tests: the two-NFP-pocket toy case from §8.4 must be found; the Lipschitz-lemma near-threshold case labels correctly in both directions; timeout ⇒ marginal), budget policy, checker-in-loop; begin `tier1`.
**Day 3:** `tier1` integration (emit PDDL problems, wrap streams, run focused/adaptive, failure decomposition, coverage audit on 30 instances), `attacks` (both targets), full `sweep` with CIs and slack binning, animation.
Acceptance tests throughout: seed-reproducibility of a full planner trace; a hand-built loose scene where every candidate is feasible and every planner solves in ≤ 2 attempts; one hand-built scene where `retrieve`-only is the Tier-1 planner's first attempted plan and fails (verifying the optimistic-enumeration behavior of §6.2); one hand-built tight scene whose only δ-packing is finger-inaccessible, which must label marginal(inaccessible) and never feasible (⟦R:M1⟧ regression test).

## 15. Scope, risks, and honest claims

**Scope.** DD2D is a 2D diagnostic instrument: it measures whether subset-selection-under-packing difficulty occurs on a naturalistic distribution and at what tightness, under an abstracted (height-stratified SE(2)×lift, floating-gripper, kinematic-clearance-grasp, quasi-static) manipulation model. It does not by itself establish 3D or real-robot significance; it gates and de-risks that investment. Publication use should pair it with a 3D flagship or frame it explicitly as the controlled tier. ⟦R:M8⟧ Positioning note for the paper: the *expected* outcome given a convex-majority library is partial or full falsification; the paper plan must be committed to either result, and a negative result from this instrument is a contribution only inside a larger paper (e.g., as the diagnostic that redirected the method/domain design), not as a standalone ICRA submission.

**Risks.** (1) *The interesting regime may be empty*: refiner adequacy (a) and shallow predictability (b) may never be simultaneously satisfied at plausible λ — that is the go/no-go, answered in one sweep; the concavity stratification tells us whether a sub-population regime exists even if the pooled one does not. (2) *Filter acceptance may be low* at loose λ (targets rarely blocked) — acceptable if reported; if F1 acceptance < 5%, widen the fill-fraction band and disclose. (3) *Checker soundness and cost*: the negative certificate is the technically hardest component (arrangement vertices + Lipschitz grid) and the schedule-critical one; day-1 fallback negatives are provisional; no attack or label-dependent numbers until the complete checker replaces it; budget timeouts land in marginal, never infeasible. (4) *Sampler confound*: headroom claims are relative to the shared refiner; report m_p sensitivity (two extra sweep cells at λ*); with accessible-packing labels, residual adequacy failures are cleanly attributable to the sampler. (5) *Tier-1 integration risk*: PDDLStream/Fast Downward setup friction, and derived-predicate (axiom) handling varies across its algorithms — validate the encoding on the hand-built acceptance scenes before trusting distribution-level numbers; if axioms misbehave under some algorithm, the pre-planned fallback is compiling the Unsafe predicates away via negated test streams (pddlstream's native facility), disclosed as an encoding variant. (6) *Coverage risk*: if the audit (§10.3) finds real planners solving substantially outside the labeled candidate space, Tier-2 claims pause until the enumerator is extended — this is a designed tripwire, not an afterthought. (7) *Budget-marginal inflation*: if labeling timeouts exceed ~10% of candidates at any λ, the confidently-labeled stratum shrinks and attack/oracle numbers lose coverage; the mitigation is raising P19, disclosed.

**Claims wording.** "On a naturalistic drawer-decluttering distribution with buffer tightness in [band], skeleton selection accounts for a measured E excess failed refinements per solve over the best cheap candidate ordering (Tier 2; call-ratio N×–M× across budgets B ∈ {100, 300, 1000}), costs an unmodified PDDLStream planner [M] stream calls per solve (Tier 1), while low-order predictors and cheap certificates reach only [x] balanced accuracy on packing feasibility" — difficulty measured on the distribution, never installed in the generator. The filters (F1–F3, with the §9.5 shift audit), placement priors (P17, settle-toward-contact), marginal-stratum handling (with reason codes), the accessible-packing label semantics, and the geometry-informed status of the Tier-2 candidate set are disclosed limitations, not hidden constraints.
