# Tote-and-Tray Decluttering (TTD): Implementation Specification

**Version:** 1.3 (consolidated). Supersedes v1.0 and the v1.1/v1.2 patches. Integrates: the
dial-consistency identity and packing-margin framework (v1.1), the staged refiner and
multi-witness composition (v1.2), the candidate-proposer honesty fixes and calibrated cost
accounting (this version), plus the remaining agreed modifications from the adversarial
review (framing/outcome tree, order-ablation evidence, compute–accuracy frontier, incumbent
fairness conditions, out-of-family shapes, statistical protocol, work packages, precision
fixes).
**Target simulator:** PyBullet (Drake port optional).
**Audience:** an implementer with TAMP/PDDLStream familiarity but no access to prior design
discussions. The document explains *why* every design element exists, then specifies *how*
to build it. Numbers marked **†** are provisional pending the §10.0 pre-pilot.

**Changelog (review item → resolution):**

| Item | Defect in v1.0 | Resolution |
|---|---|---|
| 1 | Planted nest geometrically impossible at defaults (Brunn–Minkowski; measured 328 vs 252 cm²) | §2.8 identity (C7), recomputed §3, η-based §7.3/§8.3 |
| 2 | Backjump cannot revise placements after a place failure (trace) | §9.2 staged refiner, §12.3 regression test |
| 3 | A-feas unreachable / A4 violated; single-object A4 misses pair/triple leaks (probe 3) | §8.2 multi-witness composition, A4 extended to groups ≤ 3 |
| 4 | Circular framing; no committed contribution | §1.1 grounding, §1.4 outcome tree |
| 5 | "Off-the-shelf planner" claims false; enumerator is a hidden clears-oracle | §6.2 renamed + drowning arithmetic, G1 reworded, appendix run |
| 6 | "One stream call" spans 0 to ~32 ms (probe 4: 272× spread) | §5.3 calibrated op-level accounting; dimensionless headline metric |
| 7 | Headline gap is dial-designed; no power protocol | (K−F)/(F+1) stated at G2/D1; §11 statistical protocol |
| 8 | Overclaimed impossibility language; non-exact "exact" nester | attack-relative wording, §7.2 intensified mode, §10.3 order-ablation curve |
| 9 | C5 floor tested only vs strawmen | §10.3 budgeted metaheuristic attacks + frontier requirement |
| 10 | Incumbent comparison confounded | §9.3.6 fairness conditions (modality-crossed, harnessed+native) |
| 11 | Feasible non-witness candidates never stream-verified; oracle/D3 stale under multi-witness | §8.7 verification of all feasible candidates; §9.3/§11 re-specs |
| 12 | Precision grab-bag (Gumbel score, lip garble, ε_v, σ ambiguity, A3 test, sizes, compute) | fixed in place: §5.4, §4.3.3, §5.2, §3, §8.6, §11, §12.3 |

---

## 1. Purpose and research context

### 1.1 The research gap this environment targets

Task-and-motion planning (TAMP) systems separate planning into (a) a discrete **skeleton**
(a sequence of symbolic actions with continuous parameters left free) and (b) **refinement**
(sampling/optimizing the continuous parameters — grasps, placements, motions — via
"streams"). Recent learning-for-TAMP systems accelerate this loop:

- **PIGINet** (Yang et al., 2023) learns a feasibility classifier `f(image, plan, goal) →
  [0,1]` over *complete* skeletons and reranks the candidates that a diverse symbolic
  enumerator produces. Its inputs are the initial scene image (per-object CLIP crops), goal
  literals, and the plan token sequence.
- **LAZY** (Khodeir et al., 2023) learns a goal-conditioned graph-attention policy over a
  scene graph (objects + predicates + fixed geometric features) used as a per-step prior
  inside Levin Tree Search, with online success/attempt statistics and computation-graph-key
  transfer across skeletons that share structure.

Both mechanisms represent feasibility through per-object, per-step, or low-order relational
structure. The **failure family** this project names and instruments is **subset-coupled
geometric feasibility**: skeleton feasibility governed by a global, high-interaction-order,
continuous statistic of a *set* of objects that (on the emitted distribution — see the
attack-relative discipline below) is not recovered by per-object or low-order features, has
no cheap certificate, and reveals infeasibility only late in refinement. The family occurs
in real manipulation whenever a plan must commit a set of items to a tight shared region —
stowing, decanting, container loading, dense packing (cite the packing/stowing TAMP
literature in the paper). TTD is the minimal certified instance: the discriminating
statistic is `packs(S, tray)` — joint 2D nesting of a k-object subset of concave footprints
into a tray sized only slightly larger than the subset.

Every instance is **solvable by construction** (planted witnesses with margin), refinement
of correct skeletons is easy, and an oracle told the answer solves each instance in roughly
one refinement attempt — so the benchmark measures *skeleton-selection efficiency*, not
solvability.

**Claims discipline.** Two rules govern every claim downstream. (i) *Attack-relative, not
absolute:* hardness claims have the form "attacks of interaction order ≤ j / plan-time
compute ≤ x fail on the emitted distribution", verified by the §10.3 suites — never "no
predicate set can express this" (NP-hardness of worst-case nesting says nothing about a
planted distribution). (ii) *Designed quantities are not findings:* the uninformed–oracle
gap is controlled by generator dials (§11's (K−F)/(F+1) identity); the paper reports it as
the benchmark's designed headroom, and the discoveries are the diagnostic curves (order
ablation, compute–accuracy frontier, mechanism-paired incumbent degradation).

### 1.2 Design properties the environment must satisfy

- **C1 — High-order coupling.** The discriminating statistic is `packs(S, tray)` for
  k ∈ {4,5}: an order-k continuous interaction over concave footprints with free rotations.
  Exact certification is NP-hard 2D irregular nesting in the worst case; on the emitted
  distribution, the §10.3 order-ablation suite must show low-order models near chance.
  Corollaries: the symbolic layer never encodes packing or blocking geometry (§5–6), and
  shapes are concave (§4.2) because convex packability is largely predicted by low-order
  summaries.
- **C2 — Concentrated, bimodal difficulty.** Every enumerated candidate refines with
  probability ≥ 0.9 or ≤ 0.1 (over stream seeds, within budget B). Achieved by the
  η-threshold label rule (§7.3), marginal-candidate dropping, and stream-level verification
  of every feasible candidate (§8.7).
- **C3 — No degenerate shortcut.** No universal subset; no "stage everything" plan (the
  total non-target inflated area exceeds tray capacity with margin — §3 check); no in-tote
  shuffling (§8.5 fill condition, audit A5); no sub-k clearing subset (acceptance condition
  A-size, §8.5); no stacking (§4.3.4). Search-time backtracking remains unrestricted.
- **C4 — Learnable structure.** A sufficiently expressive model (e.g., a set transformer
  over subset shape encodings) can learn the statistic; the §10.3 bracket verifies low-order
  models sit near chance while the oracle sits near 1, with the **order-ablation curve**
  (attack accuracy vs feature interaction order 1 … k−1) as the headline C1/C4 evidence.
- **C5 — Costly rejection, no cheap certificate regime.** Wrong subsets fail only at the
  last tray placement after k−1 successes. The claim is a **compute–accuracy frontier**
  (§10.3): below a stated plan-time geometric-compute level, no attack (hand-coded,
  budgeted metaheuristic, or Tier-0 learned) exceeds balanced accuracy 0.60, while exact
  checking is affordable only offline. A compute-accounted checker-in-loop baseline is a
  required comparison bracket, not a forbidden move.
- **C6 — Physical realizability within a verified window.** Quasi-static SE(2)×lift in
  PyBullet; the C2 ceiling and C5 floor must coexist. Resolved in two stages: the §10.0
  pre-pilot (pure geometry: does the η spread exceed the robustness band μ?) and the §10.2
  pilot (does the shared sampling refiner clear the ceiling?).
- **C7 — Dial consistency (new).** All dials satisfy the §2.8 identity: the Brunn–Minkowski
  necessary condition and the empirical occupancy rule Φ_f ≤ ρ̂ − h_sel. No parameter set
  enters the pilot without passing it. (v1.0's defaults violated the necessary condition
  outright.)

**Gates.** G1: the *shipped uninformed planner* (§9.3.1 — proposer + staged refiner; the
phrase "off-the-shelf" is retired, see §6.2) solves ≥ 95% at generous budget. G2: at
practical budget it burns failed refinements per solve within [5, 10] — a *designed* range:
the random-tie-break expectation is (K−F)/(F+1), e.g. 5.2 at (K,F) = (30,4). G3:
anti-leakage audits (§8.6, §10.3). G4: fair incumbent evaluation per §9.3.6. Diagnostics
D1–D5 are specified in §11.

### 1.3 Task story

An e-commerce picking cell. A storage tote arrives with one target SKU buried among rigid,
irregular items. Blockers may be staged only on a small adjacent tray — the cell's only
free surface, deliberately undersized. SKUs are fragile (no stacking) and must remain
individually re-pickable (the mandated placement clearance c_v). The operative decision
each episode: **which subset of blockers to stage**, given the chosen set must jointly fit.

### 1.4 Outcome tree (pre-registered)

| Empirical branch | Paper claim |
|---|---|
| §10.0 pre-pilot fails (η spread < μ at every operating point) | Packing coupling unworkable at these scales; switch to the pre-registered fallback (reachability/occlusion coupling); diagnosis framing unchanged |
| §10.2 pilot fails ceiling (a) after sampler escalation | Same fallback path; report the ceiling/floor incompatibility as a measured constraint on benchmark design |
| Frontier: cheap attacks fail below x ms; mid-tier set transformer succeeds | Headline: representational gap — incumbents miss subset-coupled feasibility; a set-level representation closes it at ~zero marginal plan-time cost |
| Frontier: a budgeted metaheuristic packer dominates cheaply (after k-escalation to 6–7) | Headline: measurement — fast geometric certification closes this gap class; learned per-step/whole-plan predictors structurally cannot, and the diagnosis (order-ablation + D1/D2 mechanism pairing) stands |
| A fairly-run incumbent closes the gap | Falsification reported as such; the mechanism-paired diagnostics remain the contribution |

In every branch the deliverable is the diagnosis plus the instrument; no branch leaves the
project paperless. Do not weaken pass conditions to force a branch.

---

## 2. Notation, units, and conventions

- Geometry in this document is in **centimeters**; the PyBullet implementation uses meters.
  Floor at z = 0, x–y horizontal.
- A **footprint** is a simple 2D polygon (vertex list, CCW); objects are right prisms of
  uniform height.
- **Inflation by r**: Minkowski sum with a disc of radius r (`shapely.buffer(r,
  join_style="round")`; round joins are required for the separation semantics of §2.8).
- A **stream call** is one invocation of any §5.2 row. Calls are the **control-flow** unit
  (budgets, retry caps); they are *not* a cost unit — costs are heterogeneous by ~272×
  (§5.3) and are reported per the calibrated scheme there.
- **Seeding:** every stochastic component takes an explicit integer seed, recorded in the
  instance JSON (§8.8). Identical seeds ⇒ identical traces.
- Authoritative geometry is computed in **Shapely ≥ 2.0** (version pinned; §12.3); PyBullet
  is the execution/rendering layer (physics votes only in the realism tier, §12.4).
  Rationale: C2 demands deterministic feasibility; contact simulation at millimeter scales
  would smear the labels.

### 2.8 Inflation semantics and the packing-margin radius η

Let `T` be the **raw tray interior** rectangle (W × H; no eroded region exists anywhere in
this spec). For subset S and radius r ≥ 0:

- **N(S, r) ∈ {0,1}**: 1 iff the r-inflated footprints of S admit an interior-disjoint
  placement inside `T` (free translations and rotations). Non-increasing in r. Disjoint
  r-inflated shapes have originals ≥ 2r apart; an r-inflated shape inside `T` has its
  original ≥ r from every wall — one radius encodes pairwise clearance 2r and wall
  clearance r.
- **η(S) := sup { r ≥ 0 : N(S, r) = 1 }** (sup ∅ := −∞), the *packing-margin radius*.
  Labels, decoys, validity, and the pilot are all statements about η against two
  thresholds.
- **Dials:** `c_v` — mandated placement clearance (validity-level; story in §1.3);
  **μ = 0.30 cm fixed** — robustness band = ε_s 0.15 (sampler contact back-off) + ε_v 0.05
  (numerical guard) + ε_disc 0.10 (nester rotation-grid slack).
- **Thresholds:** r_i := c_v/2 (*reachability radius* — by §5.2, the refiner can only
  realize nests at inflation r_i), r_f := c_v/2 + μ (*planting radius*).

**C7 identity.** Necessary (Brunn–Minkowski, any compact shapes):
`Σᵢ (√Aᵢ + r_f√π)² ≤ W·H`. Design rule (empirical): `Φ_f(S) := Σᵢ Ã(sᵢ, r_f)/(W·H) ≤
ρ̂ − h_sel`, where Ã(s, r) = area(s ⊕ D_r) — measured ≈ A + P·r + πr² for this family
(α = 0.98–0.99) with P ≈ 1.33 × the isoperimetric minimum — ρ̂ is the achievable
inflated-nest occupancy frontier measured in §10.0, and h_sel is the plant's selection
headroom. Φ_i is the same at r_i.

**Viability condition.** The benchmark is viable iff the subset-to-subset spread of η at
matched ΣA exceeds μ plus selection headroom — in occupancy units, spread ≳ Φ_f − Φ_i ≈
μ·ΣP/(W·H) ≈ 0.09–0.11 at the §3 operating points. [Status: spread of this order observed
under a weak constructive instrument (probe 2, repeatable 8/8); decisive measurement is
§10.0.]

The v1.0 symbol δ is retired; its duties are split across c_v, μ, and the scene-account
symbols r_s, s_tote (§8.3).

---

## 3. Parameter summary table

Starred (*) parameters are difficulty dials; † marks values provisional pending §10.0.

| ID | Parameter | Default | Notes |
|---|---|---|---|
| P1 | Tote interior (x × y) | 40 × 30 cm | walls 1.5 cm thick, 12 cm high, open top |
| P2*† | Tray interior (x × y) | 26 × 18 (OP-A) / 28 × 20 (OP-B) | lip 1.0 cm thick, 2 cm high; forced by C7 |
| P3 | Tote–tray gap | 6 cm | tray interior AABB placed 6 cm beyond the tote's outer wall |
| P4 | Object height | 6 cm | uniform |
| P5 | Library footprint area | 25–80 cm² | concave, 8–14 vertices, ≥ 1 reflex vertex |
| P5b† | Candidate-member area band | [28, 46] (OP-A) / [32, 50] (OP-B) | every object in any enumerated candidate; larger shapes are pure distractors (audit A7) |
| P6 | Objects per instance N | 12–14 | = target + member pool (n = 10) + 1–3 pure distractors |
| P7* | Witness subset size k | 4 or 5 | |
| P8*† | Φ_f occupancy band | 3 values set from ρ̂ (§10.0) | replaces v1.0's raw-area slack σ, which ignored the dominant P·r term |
| P9* | Placement clearance c_v | {1.2, 1.6} cm | μ = 0.30 fixed; r_i = c_v/2, r_f = c_v/2 + μ |
| P10 | Scramble radius r_s | 0.5 cm | scene account; tote min separation s_tote = 1.2 ≥ 2r_s + 0.2; rotation ± 4° |
| P11 | Finger width w_f | 1.5 cm | tangential |
| P12 | Finger thickness t_f | 1.0 cm | normal |
| P13 | Finger clearance c_f | 0.3 cm | column extents: tangential w_f+2c_f = 2.1, normal t_f+2c_f = 1.6; with c_v < 1.6 between-neighbor approaches are impossible and free-space approaches are verified at plant time (§8.3) |
| P14 | Gripper aperture | 0.5–14 cm | |
| P15 | Grasp descent height z_g | 3 cm | object mid-height |
| P16 | Carry height z_c | 15 cm | clears 12 cm walls |
| P17 | Antipodal tolerance | 10° | |
| P18 | Refinement budget B | 300 stream calls | per skeleton; control-flow unit only (§5.3) |
| P19 | Stage caps | t_g = 3, t_p = 5, ρ = 2 | grasp draws, pose draws per grasp, revision tokens per stage (§9.2) |
| P20* | Sampler strength m_p | 15 | pilot grid {5, 15, 40} |
| P21 | Nester rotation grid | 5° | intensified mode for infeasible certification: 1° grid, arrangement-vertex candidate points, 10× caps |
| P22 | Candidates per instance K | 28–36 | |
| P23 | Feasible candidates F | 3–5 | = number of witnesses W (supersets are infeasible by C7 arithmetic) |
| P24 | Decoys | ≥ 3 | swap-variants of witnesses; area-matched by construction |
| P25 | Feasible verification | every feasible candidate: ≥ 4/5 seeds refine within B | §8.7 |
| P26 | MI leakage threshold τ | 0.10 bits | groups of size ≤ 3 (§8.6 A4) |
| P27 | Witnesses W | 3–5 | spread rule: each object in ≤ 2 witnesses; max pairwise overlap ≤ k − 2 |

**C3 side-check at OP-B (28 × 20 = 560 cm²):** Σ_{non-target} Ã(r_i) ≈ 12 × 66 ≈ 790 cm²
≫ 560 — "stage everything" is impossible with margin. Tote capacity: Σ A_raw ≈ 520/1200 —
constructible with s_tote separations.

---

## 4. Environment definition

### 4.1 Layout

Two containers on a flat floor. The **tote** (P1) holds all N objects initially; the
**tray** (P2†) starts empty, its interior AABB placed 6 cm beyond the tote's outer wall
(P3). The tray's lip is 1.0 cm thick and 2 cm high. Because grasp descent bottoms out at
z_g = 3 cm (P15) and finger columns therefore occupy z ∈ [3, ~z_c], the 2-cm lip can never
intersect a column: **the tray lip never blocks any grasp or placement approach; tote walls
(12 cm) do block.** (This sentence replaces v1.0's garbled "tray lip-excluded regions"
line in the graspability pseudocode.) There is no eroded "usable region" anywhere in this
spec — wall clearance is carried by the inflation convention of §2.8.

### 4.2 Object shape generation

Procedural concave prisms ("polar stars"), generated per shape as:

1. Draw n ~ U{8..14} vertex angles, resampled until the minimum angular gap is ≥ 0.15 rad;
   radii ~ U[0.55, 1.0] (arbitrary pre-scale units).
2. Push 1–3 randomly chosen vertices inward by a factor U[0.35, 0.6] (creates reflex
   vertices; the polygon remains star-shaped about the origin, hence simple).
3. Scale to the target footprint area (P5 for the library at large; P5b for candidate
   members); reject if any edge < 1.0 cm, if no reflex vertex survives, or if no admissible
   antipodal edge pair exists within the aperture range (P14, P17).

Concavity is load-bearing for C1: convex packability is largely predicted by low-order
area/diameter summaries; interlocking is what makes `packs` a genuinely joint statistic.
Measured family statistics (probe 1, n = 200): perimeter ≈ 1.33 × the isoperimetric
minimum (p90 1.50); inflation gain ≈ P·r + πr² with α = 0.98–0.99.

**4.2.1 Out-of-family shape set (required split).** A secondary set built from *real object
footprints* (projected outlines of scanned household/warehouse items), area-normalized into
P5/P5b and passed through the same rejection filters. Purpose: (i) the D4 generalization
split that the procedural "held-out shapes" split cannot honestly provide (same
distribution, different seeds ≈ near-IID); (ii) external-validity anchor — the paper's
qualitative figures use these shapes so the benchmark does not read as synthetic-only.

### 4.3 Robot abstraction (core tier)

A floating parallel-jaw gripper: no arm kinematics, no IK, no base placement. `motion-ok`
is constant-true in the core tier and real in the realism tier (§12.4). Rationale: C1
requires that the *only* hard coupling is packing; arm reachability would add a second,
confounding source of infeasibility.

**4.3.1 Grasps.** A grasp of object o is an antipodal edge pair: two boundary edges whose
outward normals are anti-parallel within P17 = 10°, with face separation d ∈ [0.5, 14] cm
(P14), grasp point at the midpoint of the overlapping projection interval, fingers
descending from carry height z_c to z_g = 3 cm.

**4.3.2 Approach columns.** The swept volume of the descent, modeled as two rectangles in
the plane placed immediately outside the grasped faces at the grasp point: tangential
extent w_f + 2c_f = 2.1 cm, normal extent t_f + 2c_f = 1.6 cm (P11–P13). A grasp is
*admissible at a pose* iff both columns are free of blocking geometry there.

**4.3.3 Graspability predicate.**

```
graspable(o, scene) :=
  ∃ antipodal pair (e1, e2) of o with d ∈ [0.5, 14]:
      column(e1) and column(e2) at o's current pose intersect
      no other object footprint and no tote wall polygon
      (the tray lip never blocks — z-interval analysis, §4.1)
```

**4.3.4 No stacking; no in-tote placement.** The domain (§5.1) has exactly three actions;
there is no place-tote action, so in-tote shuffling is excluded *structurally*, and objects
are placed only on the tray floor (prisms at z = 0; stacking is not representable).
Fragility (§1.3) is the story-level justification.

---

## 5. Symbolic model, streams, and cost accounting

### 5.1 Domain (PDDL-flavored)

Types: `object`, `grasp`, `pose`. Fluents: `in-tote(o)`, `on-tray(o)`, `holding(o)`,
`handempty`, `extracted(target)`.

- `pick(o, g)`: pre `in-tote(o) ∧ handempty`; eff `holding(o) ∧ ¬in-tote(o)`.
- `place-tray(o, g, p)`: pre `holding(o)`; eff `on-tray(o) ∧ handempty ∧ ¬holding(o)`.
- `retrieve(target, g)`: pre `in-tote(target) ∧ handempty`; eff `extracted(target)`.
  Terminal action; by convention it does not set `holding` — the episode ends with the
  target extracted and hand state irrelevant (bookkeeping note, not physics).

**Rule 1 (geometry-blind symbolic layer).** No geometric literal — no `blocks-grasp`, no
`packs`, no clearance predicates — appears in the domain. A classical planner therefore
cannot set-cover its way to good skeletons. The honest corollary, stated rather than
hidden: candidate skeletons cannot come from the symbolic layer either; they come from the
shared geometric proposer of §6.2.

### 5.2 Streams

| Stream | Signature | Semantics |
|---|---|---|
| `sample-grasp` | (o) → g | draw an admissible antipodal pair (arithmetic over edge pairs) |
| `grasp-valid` | (o, g) → bool | both columns free at o's **current** pose vs all object footprints and tote walls |
| `sample-tray-pose` | (o, g) → p | compaction-biased draw, §5.4 |
| `tray-pose-valid` | (o, g, p) → bool | r_i-inflated footprint of o at p ⊆ T; distance ≥ ε_v = 0.05 to every placed r_i-inflated footprint; columns of g free at p vs placed objects (lip exempt) |
| `motion-ok` | (q, q′) → bool | constant-true (core tier); IK+RRT (realism tier) |

Implementation maintains the placed set as **r_i-inflated obstacles**. Load-bearing
consequence: the set of refiner-reachable tray configurations is, by construction, a
subset of the zero-clearance nests of r_i-inflated shapes — so the `infeasible` label of
§7.3 is sound relative to stream semantics *by construction*, not by argument. The ε_v
guard exists because touching-pose predicates are float-flaky; it is charged inside μ.

### 5.3 Budgets and cost accounting

**Control flow.** B = 300 stream calls per skeleton refinement (P18). Calls bound retries;
they are reproducible and implementation-independent. They are **not** costs.

**Why not (measured).** Probe 4, at the operating point (N = 12, m_p = 15, ≤ 4 placed):

| stream call | est. cost | vs `grasp-valid` |
|---|---|---|
| `motion-ok` | ~0 | 0× |
| `sample-grasp` | ~2 µs | ~0× |
| `tray-pose-valid` | ~0.05 ms | 0.4× |
| `grasp-valid` | ~0.12 ms | 1× |
| `sample-tray-pose` (m_p = 15, uncached NFPs) | ~32 ms | **272×** |

Underlying primitives: prepared predicate ~4 µs; candidate construct+test ~29 µs; buffer
~42 µs; one NFP construction ~6.4 ms. NFP caching can compress the sampler's cost by up to
~70× in the best case, so per-call costs must be *measured on the instrumented core*, never
assumed.

**Accounting scheme.** The 2D core instruments op-level counters: **P-ops** (prepared
predicate evaluations) and **C-ops** by kind (polygon construction, buffer, NFP/IFP
construction, union). `calibrate.py` measures µs/op on pinned library versions and
reference hardware and publishes the calibration table. Every experiment reports the
**cost triplet**:

1. **Headline (dimensionless):** solve rate and **failed refinements per solve** — robust
   to hardware, implementation, and caching; this is the number in the abstract.
2. **Calibrated geometric cost:** dot product of op counts with the calibration table,
   *plus* the raw per-type counts so third parties can re-weight.
3. **Wall-clock,** hardware disclosed.

**Checker-in-loop commensurability.** The §7.2 nester instruments the same counters (it is
built from the same op classes; a 2 s anytime cap ≈ 314 NFP-equivalents, measured), so the
checker-in-loop baseline is compared in the same currency. The v1.0 "equivalent mean stream
call" conversion is retired.

### 5.4 Compaction-biased placement sampler

Per call: draw θ from the 5° grid with uniform jitter; with probability 0.7 generate
**contact proposals** — points on the NFP/IFP boundaries of the r_i-inflated obstacle set,
backed off by ε_s = 0.15 cm along the separating normal — else uniform proposals
(exploration); take m_p proposals total and return the minimizer of the scalar score

```
score(x, y) = y + 0.02·x + G,   G ~ Gumbel(0, β = 0.3 cm)
```

(bottom-left compaction with tie-noise; this replaces v1.0's type-incoherent "Gumbel noise
on a lexicographic preference"). Contact proposals sit exactly on the validity boundary +
ε_s, which is why they can realize snug nests. This bias is **load-bearing, not an
optimization**: probe 2 measured that uniform free-space sampling saturates near inflated
occupancy 0.6 and a 20×-compute version of it upgraded 0/6 failures — without contact
proposals the C2 ceiling is unreachable at the §3 operating points.

---

## 6. Skeleton space and the geometric candidate proposer

### 6.1 Skeletons

One skeleton per candidate subset S: `[stage(o) for o in π(S)] + retrieve(target)`, where
`stage(o) = pick(o); place-tray(o)` (the refiner's macro-step, §9.2) and π(S) is the
candidate's **published greedy peel order** (§8.6, A1′) — the deterministic order obtained
by repeatedly removing any member with a currently free grasp. π replaces v1.0's
ascending-object-ID convention because subset members crowd the target and can block each
other's tote grasps; a geometry-blind order could fail at a mid-sequence pick, which would
(i) make some feasible candidates unrefinable under the published order and (ii) produce
*early* failures that pollute the D2 late-failure profile. π is derivable from public scene
geometry, so publishing it leaks nothing a solver could not compute.

### 6.2 The geometric candidate proposer (shared clears-oracle)

Renamed from v1.0's "enumerator" to say what it is. The proposer computes the target's
grasp corridors and each corridor's **blocker set** from finger-column geometry — i.e., it
solves the `clears` half of `feasible(S) = clears(S) ∧ packs(S)` and hands the result,
identically, to every evaluated method. **TTD deliberately gives `clears` away to isolate
`packs`.** This is a design choice stated in the paper, not an implementation detail.

Why candidates cannot come from a symbolic planner (arithmetic, verified once in an
appendix run): the optimistic symbolic layer accepts *any* alternating pick/place prefix +
retrieve, so ascending-length diverse enumeration over the n = 10 member pool must exhaust
Σ_{s<k} C(10, s) sub-clearing skeletons before the first size-k candidate — 386 skeletons
at k = 5 (176 at k = 4). With acceptance condition **A-size** (§8.5: every corridor's
blocker set has ≥ k members — v1.0 asserted this "structural fact" without enforcing it),
each sub-clearing skeleton binds its stages cheaply and then fails only at `retrieve`,
burning ≈ B; total ≈ 386 × ~300 ≈ 116k calls before any candidate that can succeed, versus
≈ 1.6k for the shipped planner — a ~70× tax (~26× at k = 4). Gate G1 is therefore defined
relative to the **shipped uninformed planner** (proposer + staged refiner); the phrase
"off-the-shelf planner" is retired throughout.

**Proposer output.** Base candidates: the distinct minimal blocker sets across corridor
clusters. Padding: supersets that add the **largest** adjacent non-member (the occupancy
increment ΔΦ ≈ [Ã(e, r_i) − (μ·Σ_w P + kπ(r_f² − r_i²))]/(W·H) is then comfortably
positive, landing the superset cleanly infeasible rather than marginal), until
K ∈ [28, 36]. Marginal candidates are dropped per §7.3. Fairness scope note for §9.3.6:
PIGINet's home pipeline draws diversity from a symbolic planner; on TTD all methods rank
the shared proposer's list — a stated re-implementation condition of G4.

### 6.3 Order-irrelevance (tray side) — verified, not asserted

Claim: for a planted witness, any insertion order of its members refines. Sufficient
condition, checked at plant time (§8.3, step 1c): every witness pose admits ≥ 1 admissible
grasp whose columns avoid **all** other witness footprints at their planted poses — columns
free of all members ⇒ free of any prefix. Status: a per-instance verified property (the
plant rejects nests failing it), with audit A6 (3 random orders × 2 seeds on the primary
witness) as the end-to-end empirical check that the *sampler* also realizes it.

---

## 7. The nester and the label rule

### 7.1 NFP machinery

No-fit polygons via convex decomposition + pairwise convex Minkowski sums + union;
inner-fit polygon (IFP) for the container. Candidate placements live on the NFP/IFP
arrangement. This machinery is shared by the label-time nester, the §5.4 sampler's contact
proposals, and (instrumented identically, §5.3) the checker-in-loop baseline.

### 7.2 The nester (exact-in-intent, honestly discretized)

Depth-first search over discretized candidate poses: rotations on the 5° grid (P21),
positions at NFP/IFP boundary vertices and edge midpoints; random-restart with random grid
offsets; per-subset node caps; an **anytime mode** (budget-capped best-effort, used by the
checker-in-loop baseline and generation-time triage). **Intensified mode** (used for every
`infeasible` certification): 1° rotation grid, candidate positions extended to arrangement
vertices (NFP–NFP intersections), 10× node caps.

Honesty note (replaces v1.0's "provably fail to nest"): the search is complete only with
respect to its discretization. On the feasible side the μ band absorbs discretization
error (a certificate at r_f survives grid-scale perturbation). On the infeasible side a
razor-thin missed nest is a one-sided error — harmless to the *stream-level* bimodality C2
actually operationalizes, because a measure-zero-thin nest is unreachable by the sampling
refiner; the spec's language is "checker-certified (intensified)", never "provable".

### 7.3 Label rule

For each enumerated candidate S, with η, r_i, r_f from §2.8:

- **feasible(S)** ⇔ η(S) ≥ r_f — certified constructively by a stored nest of the
  r_f-inflated shapes.
- **infeasible(S)** ⇔ η(S) < r_i — certified by intensified-mode exhaustion at r_i.
- **marginal** (r_i ≤ η < r_f): dropped from the candidate set (not the instance). If
  > 20% of enumerated candidates are marginal, or any *witness* is marginal, regenerate.
  Rationale: marginal candidates are exactly those whose refinement is seed-dependent
  inside the μ band; v1.0's regenerate-on-any-marginal combined with superset padding
  would thrash. Out-of-generator instances receive no such screening (consistent with the
  rankings-only claim on that split).

Soundness: feasible ⇒ a fat target for the shared refiner (μ headroom over the validity
radius r_i); infeasible ⇒ unreachable under §5.2 semantics. §8.7 verifies the ≥ 0.9 / ≤ 0.1
stream-level consequence for **every feasible candidate** (P25) and spot-checks the
infeasible side.

---

## 8. Instance generator

### 8.1 Flow

member pool → **composition pattern** (combinatorial, §8.2) → **witness planting**
(geometry, §8.3) → **occlusion build** (§8.4) → decoys/distractors + repair (§8.5) →
audits (§8.6) → scramble + verification (§8.7) → emit (§8.8). Expensive geometry is never
spent on a composition the counting stage has not already approved.

### 8.2 Composition pattern (before any geometry)

Fix the candidate-member pool M, |M| = n = 10, all areas in P5b. Choose W ∈ [3, 5] witness
index sets of size k satisfying the **spread rule** (P27): every object in ≤ 2 witnesses;
max pairwise overlap ≤ k − 2. (Existence: for W = 4, k = 5, n = 10 an exactly 2-regular
design exists, e.g. {0..4}, {5..9}, {0,1,2,5,6}, {3,4,7,8,9}.) Project the proposer's
expected additions (2 supersets per witness, ≥ 3 decoy swaps, filler covers to K) and run
the **composition pre-check**: exact plug-in MI of every single, pair, and triple of
member objects against the planned labels ≤ 0.7·τ (slack left for geometric realization).
Counting only — C(10,2) = 45 pairs, C(10,3) = 120 triples, microseconds.

Evidence this is both satisfiable and necessary (probe 3; MI in bits, τ = 0.10, K = 30,
F = 4): balanced design — max single 0.005, pair 0.075, triple 0.052 (≈ 20× slack);
near-duplicate design (witnesses sharing a 4-core, i.e., what a lazy repair loop would
produce) — singles pass at 0.096 while pairs/triples leak at **0.148 / 0.181**. Auditing
only single objects, as v1.0 did, certifies datasets a pairwise-featurized Tier-0 attacker
can read the plant from; hence A4's group extension (§8.6).

### 8.3 Witness planting

Per witness Wᵢ: draw its not-yet-assigned members' shapes from P5b subject to
Φ_f(Wᵢ) ∈ [Φ_lo, Φ_hi]†; run the nester at inflation r_f; on failure resample free members
(≤ 50 tries per witness, then RESTART — this selection is h_sel, deliberately harvesting
the high-complementarity tail of η). **Step 1c (column check):** every member's planted
pose must admit ≥ 1 admissible grasp whose columns avoid all other members' planted
footprints (lip exempt); failures count toward the 50 tries. Witnesses are *alternative*
plans: each must pack on its own; no simultaneous packing is required. The primary witness
W₁ is the first planted; its certificate is the oracle's answer (§9.3.2).

**Margin accounts** (two ledgers; v1.0's single table both over-allocated — 2.5 > 2.4 —
and charged tote-side scramble against a tray-side certificate it cannot touch):

| Account | Symbols | Pays for | Never charged here |
|---|---|---|---|
| Nest (tray) | μ = ε_s 0.15 + ε_v 0.05 + ε_disc 0.10 | sampler back-off, validity numerics, rotation-grid slack | scramble (η depends only on shapes); finger columns (explicit check 1c) |
| Scene (tote) | r_s = 0.5, s_tote = 1.2, corridor inflation 0.5 | scramble survival of tote separations and corridor structure | nest geometry |

### 8.4 Occlusion build

Assign each witness a target grasp corridor (c* to W₁). Place members in the tote,
s_tote-separated, so that each corridor cᵢ's blocker set is **exactly** Wᵢ (shared members
intersect both assigned corridors' columns — geometrically natural, corridors of one target
are angularly adjacent); every remaining corridor is blocked by distractors; removing any
single Wᵢ frees its corridor with 0.5 cm column-inflation margin (audit A1).

### 8.5 Decoys, distractors, repair loop

- **Decoys (≥ 3, P24):** swap-variants of witnesses — exchange 2 members for 2 non-member
  pool objects (area-matched by P5b construction, so the raw summed-area attack H1 is
  near-chance by design) — certified infeasible at r_i in intensified mode.
- **Distractors:** 1–3 objects outside M (any P5 area, including large): block secondary
  corridors, enforce **A-size** (every corridor's blocker set ≥ k — the acceptance
  condition v1.0 lacked; without it a small blocker set on a secondary corridor is a
  trivially feasible sub-k candidate), and fill free pockets near the target (with the
  structural fact that no place-tote action exists, this closes in-tote maneuvering).
- **Repair loop:** on any acceptance failure, apply local repairs (move/resize a
  distractor, swap a decoy member, re-draw a filler cover). **Ordering rule:** any repair
  that changes candidate membership re-runs the §8.2 composition pre-check *before* any
  nester call. Acceptance conditions: A-count (K ∈ [28, 36]); A-feas (F = W ∈ [3, 5], each
  with a stored r_f certificate); A-size; A-decoy (≥ 3); A-marg (≤ 20% dropped, no marginal
  witness); A-olap (every witness object in ≥ 1 infeasible candidate; every decoy object in
  ≥ 1 other candidate). Bounded loop; RESTART after 25 iterations.

### 8.6 Audits (dataset- and instance-level)

- **A1 — clearing margins:** each candidate, after removal of its members, frees its
  corridor under 0.5 cm column inflation.
- **A1′ — peel executability:** every enumerated candidate admits a complete greedy peel
  order (deterministic geometric check); the order is published in the JSON. [Surfaced
  while formalizing §9.2; frequency of mutual grasp-blocking untested, check is ~free.]
- **A2 — initial hardness:** the target is ungraspable in the initial scene (every
  corridor blocked).
- **A3 — plant detection:** the Tier-0 suite's average precision at ranking each
  instance's candidates by predicted feasibility is statistically indistinguishable from
  the base rate F/K (permutation test, α = 0.05, batches of 200 instances).
- **A4 — group leakage:** exact plug-in MI between X_G = [G ⊆ S] and the label, over the
  instance's candidates, ≤ τ = 0.10 bits for every member group |G| ≤ 3; no one-sided
  perfect predictor. Finite-sample note: plug-in MI at K ≈ 30 is biased upward ≈ 1/(2K ln 2)
  ≈ 0.024 bits — conservative for an upper-bound audit; bias-corrected values reported
  alongside. Orders > 3 are covered at dataset level by the §10.3 learned attacks.
- **A6 — order robustness:** the primary witness refines under 3 random member orders × 2
  seeds.
- **A7 — member-area matching:** two-sample KS (α = 0.05, per batch) between areas of
  objects appearing in ≥ 1 feasible candidate vs objects appearing only in infeasible
  candidates. (Witness sizing restricts witness-member areas; without this audit,
  per-object area is a plant-detection feature.)

(v1.0's A5 "no-shuffle" audit is absorbed by A2 + A-size + the structural absence of
place-tote.)

### 8.7 Scramble and verification

Scramble every tote pose (translation ≤ r_s = 0.5 cm, rotation ≤ 4°), then re-run A1, A1′,
A2 and re-derive peel orders. Scramble cannot change any η (labels depend only on shapes)
— stated to kill v1.0's category error permanently. Then **stream-verify every
feasible-labeled candidate**: ≥ 4/5 seeds refine within B under the shipped refiner (P25);
downgrade-and-repair or regenerate on failure. Spot-check the infeasible side (3 candidates
× 2 seeds, require 0/6 successes). The feasible-side verification is the empirical half of
C2; the infeasible side is carried by construction (§5.2) plus the spot check.

### 8.8 Instance record and splits

JSON per instance: seed chain; shapes (vertex lists) and poses; member pool and P5b band;
candidate list with labels, peel orders, Φ_f/Φ_i values, and (for feasible candidates)
nest certificates; witness list with primary flag; audit results (including per-instance
A4 group maxima); generator version hash.

Splits: train ≥ 5,000; val 500; test 500; held-out-shapes 500 (same family, disjoint shape
seeds — reported as near-IID, which it is); **out-of-family 300** (§4.2.1 real footprints);
out-of-generator 100 (hand-built scenes, rankings-only claims); N-extrapolation set (N →
1.5N, candidate counts re-derived). Generation compute: labeling dominates (K ≈ 30 nester
labels/instance, intensified certification on the infeasible side); budget ≈ 2–5 CPU-min
per instance ⇒ ~10–25 CPU-days per 5k instances before parallelization — plan a cluster
run; the §10.0 pre-pilot doubles as the cost-model measurement.

---

## 9. Planner, refiner, and evaluated variants

### 9.1 Shared architecture

Every evaluated method is the same pipeline — proposer (§6.2) → skeleton selection policy →
staged refiner (§9.2) — differing **only** in the selection policy (the order/priority over
the shared candidate list). This is what makes cost comparisons attributable to skeleton
selection and nothing else.

### 9.2 Staged refiner with revision tokens

*(Replaces v1.0 §9.2, whose one-step backjump could never revise an earlier placement
after a placement failure: the step before a failed place is that object's own pick, which
re-binds on the first draw — the tote is unchanged — so its failure counter never reached
the cap and control ping-ponged between re-grasp and re-fail until B. Cycle arithmetic:
~10 place failures (20 calls) + 1 pick re-bind (2 calls) ≈ 22 calls/cycle, ~13 cycles to
B = 300, earlier placements untouched. Placement revision was reachable only from
pick/retrieve failures — not from the placement failures that matter for the C2 ceiling
and the D2 evidence.)*

**Macro-steps.** The unit of binding and backjumping is the **stage**: `stage(o) :=
pick(o); place-tray(o)`, bound jointly as (g, p); `retrieve` is stage k+1 (grasp only).
Rationale: pick and place of one object are coupled through the approach columns (the
grasp fixes which placements validate), and the v1.0 defect was an artifact of splitting
them at the backjump level.

**Within a stage visit:** up to t_g = 3 grasp draws (2 calls each); each valid grasp gets
up to t_p = 5 tray-pose draws (2 calls each, validity per §5.2 with that grasp's columns);
first valid (g, p) binds (+1 motion-ok). Worst-case visit: 3·2 + 3·5·2 = 36 calls. Grasp
re-sampling is the cheap first resort for column-caused placement failures — it changes
the approach constraint without disturbing the tray.

**Revision tokens.** Each stage carries ρ = 2 monotone tokens (never reset): the number of
times its committed binding may be discarded because a *later* stage failed.

```
refine(skeleton, seed, B):
    tokens[1..k] = ρ;  i = 1;  calls = 0
    while calls < B:
        if visit(stage_i) == BOUND:                # t_g × t_p ladder
            i += 1
            if i > k+1: return SUCCESS(bindings, calls)
        else:
            m = max { m < i : tokens[m] > 0 }
            if m exists:
                tokens[m] -= 1
                undo stages m..i−1                 # suffix undo; objects return (logically)
                i = m                              #   to their original tote poses
            # else: re-visit stage i until B exhausts (v1.0's step-0 rule, preserved:
            #   the full-budget burn on infeasible subsets IS C5's costly rejection)
    return FAILURE(calls = B)
```

**Suffix undo, not point undo.** Undoing the contiguous suffix m..i−1 and rebuilding in
order preserves the invariant that every committed binding was validated against exactly
the geometric state that will hold at its execution time. Point-revising stage m under
kept later stages breaks it twice: o_m's new pose gets validated against objects not yet
on the tray at its execution step, and the later objects' column checks become stale
against o_m's old pose. Backjumping remains **search-time** bookkeeping (v1.0 C3's
distinction unchanged); no re-grasp feasibility is charged for an undo.

**Accounting sanity check.** Planted subsets: grasps typically valid in 1–2 draws, poses
in 1–3 (the μ margin is a fat target) ⇒ ~7–11 calls/stage ⇒ k = 5 plus retrieve ≈ 45–90
calls with a few retries — the v1.0 oracle bracket (~50–150) stands. Infeasible subsets:
stages 1..k−1 bind cheaply; stage k alternates 36-call visits with suffix rebuilds (~10–40
calls); B = 300 exhausts after ~4–6 last-stage cycles with deepest-stage = k — the
late-failure signature, now produced by a mechanism that genuinely could have revised
placements along the way.

### 9.3 Evaluated variants

1. **Uninformed (shipped planner; defines G1/G2):** candidates in proposer order
   (ascending |S|, random tie-break). Expected failed refinements under random tie-break =
   (K − F)/(F + 1) — reported next to every gap number.
2. **Planted oracle:** reads the **primary** witness and its certificate from the JSON and
   refines only it. With multiple witnesses, an argmin-cost-over-witnesses oracle would be
   strictly stronger; the primary-witness convention keeps the bracket a *no-search* upper
   anchor with v1.0 semantics. The per-instance F distribution is reported alongside.
3. **Checker-in-loop:** run the anytime nester (2 s cap†) on candidates in proposer order;
   refine the first survivor. Costs in the shared op currency (§5.3). This is a *required
   bracket*: if it dominates the frontier, that is a finding (§1.4), not an embarrassment
   to be designed away.
4. **Retrieval baseline:** nearest-neighbor over (shape-set, tray) signatures from the
   training split, propose the stored witness of the closest instance. Role per §11 D3.
5. **Mid-tier learned selector:** set transformer over per-object shape encodings of each
   candidate (+ tray dims), trained on generator labels, used as a reranker. This is the
   C4 demonstration and the paper's method exemplar.
6. **Incumbent re-implementations (gate G4).** Fairness conditions, all mandatory:
   trained on the same training split at matched scale with tuning budgets logged; same
   candidate lists; sanity-checked on a home-turf task before transfer claims. PIGINet
   style: transformer over (rendered per-object crops, plan tokens, goal) — plus, if it
   underperforms the mid-tier, a **modality-crossed** variant (same architecture, exact
   shape encodings) to separate architecture from input privilege before any
   representational claim. LAZY style: **both** harnessed (GAT policy reranking the shared
   list — controlled) and native (its own lazy search over the TTD domain with the shared
   streams — faithful), reported separately. Mechanism-paired diagnostics: D1 for
   rerankers, D2 for online-statistics methods.

---

## 10. Pre-pilot, pilot, and attack suites

### 10.0 Pre-pilot: measure the η landscape (pure geometry; first experiment after the nester exists)

For each operating point (OP-A, OP-B) × 3 Φ_f bands: sample 200 random k-subsets from the
member band at matched ΣA; compute η by bisection (3–4 nester calls each; instrument the
op counters — this run doubles as the §5.3 cost model and the labeling-cost estimate).
Read off: the frontier ρ̂; the η spread at matched ΣA; witness supply (tail mass η ≥ r_f);
decoy supply (mass η < r_i); band occupancy (mass in [r_i, r_f)).

**Go criteria:** witness supply ≥ 10% (raised from 5% to cover W-fold selection), decoy
supply ≥ 10%, band occupancy ≤ 15%, at some cell. The passing cell fixes P2/P5b/P8 and the
pilot grid. **No-go:** the §2.8 viability condition fails empirically → the pre-registered
fallback (reachability/occlusion coupling) triggers *before* the generator, refiner
integration, or any learning code is written.

### 10.1 Dials and 10.2 pilot

Dials: c_v ∈ {1.2, 1.6} × Φ_f band (3 values from §10.0) × m_p ∈ {5, 15, 40} — 18 cells,
each pre-screened by C7. Per cell, 200 instances; pass conditions: (a) ceiling — every
witness refines ≥ 0.9 (this is where the staged refiner and contact sampler are actually
tested); (b) floor — every infeasible candidate ≤ 0.1; (c) gates G1/G2. Grid extension, if
needed, moves along Φ_f (the variable the frontier arithmetic controls). The pilot report
(pass/fail per cell, with the D1/D2 distributions) ships in the paper — pilot results are
evidence, not internal bookkeeping.

### 10.3 Attack suites and the compute–accuracy frontier

**Hand-coded (H):** H1 summed raw area vs tray; H2 greedy bottom-left insertion (single
order); H3 pairwise min-NFP-clearance aggregate; H4 largest-object/diameter heuristics.
**H5 (new): budgeted metaheuristic packers** — simulated annealing nester at 50 / 200 /
1000 ms plus one public nesting library at matched budgets. These occupy the previously
unexplored regime between microsecond heuristics and the 2 s anytime-exact checker, and
they are the attack a reviewer will propose first.

**Tier-0 learned:** logistic/GBM on low-order features — order-1 (per-object areas,
diameters, counts), order-2 (pairwise NFP/overlap summaries), order-3 (triplet nesting
bits), and **(k−1)-subset nesting bits** (closing the order-4 hole at k = 5). Reported as
the **order-ablation curve**: balanced accuracy vs feature interaction order 1 … k−1. This
curve is the headline C1/C4 evidence — a slow climb that saturates only at order k is the
claim, made attack-relative.

**Frontier requirement (replaces v1.0's binary C5 floor):** report balanced accuracy vs
plan-time geometric compute (calibrated units) for all attacks, the checker-in-loop
curve, and the trained mid-tier selector (~zero marginal plan-time cost, training
amortized and disclosed). The benchmark's validity claim: **there exists a compute level
x such that every attack below x sits ≤ 0.60 balanced accuracy while the learned selector
exceeds its target above it.** Pre-registered escalation if H5 dominates: k → 6, 7 (nesting
cost grows super-linearly); if no window exists even then, the honest branch of §1.4
applies.

---

## 11. Metrics, diagnostics, and statistical protocol

**Headline metrics:** solve rate; **failed refinements per solve** (dimensionless — the
abstract number); the §5.3 cost triplet as supporting evidence.

- **D1 (rerankers):** rank of the first feasible candidate under each policy, reported
  against the per-instance analytic random-order expectation (K − F)/(F + 1) — so designed
  headroom and learned gains are never conflated.
- **D2 (refinement traces):** per rejected skeleton — deepest stage reached, revisions
  consumed by stage, calls, op counts. The late-failure histogram is the C5 evidence and
  the mechanism-pairing for online-statistics methods (LAZY-style CG keys mis-assign blame
  when failures concentrate at the last stage of a jointly-infeasible set).
- **D3 (retrieval):** evaluated on held-out-shapes and out-of-family splits, where
  shape-similarity transfer should fail — that failure is the anti-memoization evidence.
  On the IID split retrieval finding *any* witness is a legitimate solve (multi-witness),
  so IID retrieval is reported as a dataset-diversity diagnostic only.
- **D4 (generalization):** IID → held-out-shapes (near-IID; labeled as such) →
  out-of-family (the real test) → N-extrapolation.
- **D5 (execution):** realism-tier execution success and ranking preservation (§12.4).

**Statistical protocol:** test split ≥ 500 instances; paired per-instance comparisons with
bootstrap CIs (10k resamples); ≥ 3 training seeds for every learned method (seed spread
reported); mean, median, and P90 of per-solve cost. No headline claim without its CI.

---

## 12. Software architecture, build order, and work packages

### 12.1–12.2 Architecture and modules

`ttd_core/` (authoritative 2D geometry: shapes, columns, NFP/IFP, nester, streams — all op
counters live here), `ttd_gen/` (composition, planting, audits, JSON), `ttd_plan/`
(proposer, staged refiner, variants), `ttd_learn/` (Tier-0 attacks, mid-tier, incumbent
re-implementations), `ttd_sim/` (PyBullet build + realism tier), `ttd_eval/` (metrics,
frontier plots), plus `calibrate.py` (publishes the §5.3 cost table) and
`verify_labels.py` (re-derives every label of a released dataset from the pinned geometry
stack — the reproducibility escrow).

### 12.3 Build order (with the two mandatory checkpoints)

1. `ttd_core` shapes + NFP + nester (unit tests: hand-made feasible/infeasible nests;
   **revision-required regression test** — a pinned-seed 3-object scene solvable only by
   revising the first placement after a placement failure; assert the staged refiner
   succeeds with ≥ 1 placement revision in-trace AND a tokens = 0 control, behaviorally
   equivalent to the v1.0 defect, fails).
2. **§10.0 pre-pilot** — go/no-go before any further code.
3. Streams + sampler + staged refiner integration; 4. generator + audits; 5. **§10.2
   pilot**; 6. dataset generation (cluster run; compute estimate §8.8); 7. attacks +
   mid-tier + incumbents; 8. realism tier. Versions pinned (Shapely/GEOS, PyBullet,
   NumPy); every released dataset ships with `verify_labels.py` output.

### 12.4 Realism tier (required paper deliverable, not optional)

Franka (or equivalent) arm in PyBullet: real IK/RRT `motion-ok`, physics-stepped
pick/place, same 2D core as ground truth. Deliverable: execution success of core-tier
plans and **ranking preservation** (does the learned selector's candidate ordering survive
the realism tier?) on ≥ 100 instances, one figure with §4.2.1 real-footprint shapes.
Purpose: closes the "toy 2.5D" review without letting physics vote on labels.

### 12.5 Work packages

**P0** (go/no-go): core geometry, nester, dial-consistency computation, staged refiner,
pre-pilot. **P1** (the ICRA paper): generator + dataset, attack suites + order-ablation,
brackets (uninformed/oracle/checker-in-loop/retrieval), mid-tier selector, one faithful
configuration each of PIGINet-style and LAZY-style (harnessed), realism-tier deliverable.
**P2** (post-submission): full incumbent factorial (modality-crossed, native LAZY),
k-escalation, composed clears∧packs tier. The paper is P0 + P1.

---

## 13. Video pipeline

Per-episode renders from the PyBullet build: candidate overlay (predicted feasibility per
method), refinement trace with stage markers and revision events (the D2 money figure —
k−1 clean placements, then the last object hunting for a pocket that does not exist), and
side-by-side selector comparisons. Real-footprint shapes (§4.2.1) in all paper-facing
renders.

## 14. Risk register (updated)

| Risk | Status / mitigation |
|---|---|
| η spread < μ (viability condition fails) | The central empirical risk; measured decisively and cheaply at §10.0 before major build-out; pre-registered fallback |
| 3–5 witnesses per instance unrealizable at sane RESTART rates | Hypothesis; §10.0 witness-supply criterion raised to 10%; spread rule keeps witnesses membership-diverse, not geometry-identical |
| Ceiling (a) fails at pilot despite pre-pilot pass | Sampler escalation path (m_p, contact ratio) is a dial, then fallback; the staged refiner's regression test de-risks the known failure mode |
| H5 packers dominate the frontier | Pre-registered k-escalation; honest branch of §1.4 remains publishable (diagnosis + frontier measurement) |
| Incumbents close the gap when run fairly | Reported as falsification; mechanism-paired diagnostics remain the contribution |
| All claims are enumeration-relative (shared proposer) | Owned in §6.2 with the ~70× drowning arithmetic and an appendix verification run |
| Generation compute | Estimated §8.8; measured at §10.0; anytime triage before intensified certification |

## 15. Quick start for the implementer

Read §2.8, §3, §7.3, §9.2 first — the packing-margin radius η, the two thresholds, and the
staged refiner are the load-bearing structures; everything else is machinery around them.
The three delicate components, in build order: the nester (intensified mode is what makes
`infeasible` labels honest), the composition pre-check (counting before geometry — never
let the repair loop discover combinatorial impossibilities with nester calls), and the
contact-proposal sampler (probe 2: uniform sampling saturates at occupancy ~0.6; the C2
ceiling is unreachable without contact proposals). Run the pre-pilot before writing
anything downstream of it.
