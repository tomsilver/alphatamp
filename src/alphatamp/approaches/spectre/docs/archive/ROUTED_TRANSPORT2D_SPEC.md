# RoutedTransport2D: Environment Specification (v1)

**A multi-axis-latent, tag-augmented substrate for test-time adaptive skeleton reordering research**

_Version 1.0 — pre-implementation specification, intended to be read alongside `SPECTRE_METHOD_SPEC.md`. Sections marked_ ⚠ _contain decisions that are provisional pending milestone validation._

_Note on scope: this document specifies the environment and its integration contract with SPECTRE. It does not re-specify SPECTRE itself; for the method architecture, loss, and training objective, defer to `SPECTRE_METHOD_SPEC.md`._

_Note on the training pipeline spec: this document references `SPECTRE_TRAINING_PIPELINE_SPEC.md` where relevant but is written to be consistent with `SPECTRE_METHOD_SPEC.md` §5 in its absence. Claude Code: if the training pipeline spec is available, its F-subset sampling schema and episode-record format take precedence over §6.3 and §6.4 below._

_Note on version history: v1 supersedes the prior v0 design (single-axis color-passage latent on K₃,₃ topology). The v1 design preserves v0's structural-guarantee approach to defeating universally-safe skeletons, but adds two further mechanisms — a second latent axis (grasp mode) and per-problem static tags (passage widths and item sizes) — needed to defeat the discrete-key adaptive baseline B4 (per the SPECTRE EDA spec §4.4). v0 is documented inline where useful for contrast; the v0 file should be considered deprecated._

---

## 1. Purpose and motivation

### 1.1 What v0 got right and what v0 missed

The v0 design committed to two structural properties of the substrate:

(P-S1) _No universally-safe skeleton._ The K₃,₃ topology with same-side source/target instances guaranteed that every candidate skeleton uses exactly two of three passage colors in its loaded traversals. Combined with a per-scene `blocked_color` latent, every skeleton was vulnerable to at least one mode and committed to one of three anti-correlated families.

(P-S2) _Multiple structurally-similar skeletons per family._ For N=3, each family contained 6 skeletons (one per item ordering). The historical baseline (B3 in EDA terminology — see SPECTRE EDA spec §4.3) wastes attempts cycling through all 6 same-family variants before switching families, while an adaptive method switches after one observation.

(P-S1) and (P-S2) together produce a 2.75-attempt B3-vs-optimal premium at N=3 — but this premium is also fully recoverable by **B4**, the adaptive Naive-Bayes log-odds baseline (EDA spec §4.4). B4 maintains pair-conditional success estimates p̂(k | k' failed) and updates rankings online via log-ratio sums. On v0, B4's pair table fits cleanly from 500 episodes (only ~24 canonical keys; every (k, k') co-occurs in hundreds of training problems), and B4 saturates the same Bayes-optimal policy SPECTRE is targeting. The B3-vs-B4 adaptive premium = 2.75 attempts; the B4-vs-SPECTRE gap = 0.

This is the gap v1 is designed to close. The B4 ceiling is real: any environment with a small-discrete-latent, well-separated-family structure is naturally solved by Naive-Bayes failure-conditioning. SPECTRE's architectural advantage — continuous skeleton embeddings and per-failure attention pooling — only matters when the latent space exceeds B4's discrete-pair-table capacity, or when the failure-observation signal is _confounded_ in a way only continuous representations can disentangle.

### 1.2 The two mechanisms added in v1

**Mechanism A: Multi-axis latent.** v1 splits the per-scene latent into two independent axes — `blocked_color ∈ {c₁, c₂, c₃}` (as in v0) and `blocked_grasp ∈ {top, side}` — giving 6 modes total. Skeletons commit to a (color_pair, grasp) pair, partitioning into 6 anti-correlated families. The per-family marginal success rates span a wider range (highest 0.30 down to lowest 0.08 under default priors), which amplifies B3's waste and grows the per-episode candidate pool from 18 to 36 (capped at 30).

This alone does not break B4 — pair-table density is still adequate at 500 episodes — but it is the prerequisite for mechanism B to deliver a meaningful gap. With only 3 modes (v0), the residual confounding from per-problem tags still fits within B4's per-pair sample size; with 6 modes, the marginal pair-table cells become small enough that confounding meaningfully degrades B4's posterior estimates.

**Mechanism B: Per-problem static tags.** Every problem instance carries two sets of static atoms in its initial abstract state s₀:

- `PassageWidth(passage, w)` for each passage, with `w ∈ {narrow, medium, wide}` sampled per-passage per-problem.
- `ItemSize(item, s)` for each item, with `s ∈ {small, medium, large}` sampled per-item per-problem.

Refinement requires _per-traversal size-fit_: a `TraverseLoaded` op with item `i` through passage `p` succeeds only if `size(i) ≤ width(p)` under a fixed total order. This is checked at refinement time only — the _symbolic planner does not condition on the tags_. The candidate pool returned by the planner is therefore tag-independent (always 36 skeletons before capping); but each candidate's true per-problem success rate depends on whether its specific traversals are size-compatible.

The mechanism by which this defeats memorization is: B4's pair-conditional p̂(k | k' failed) becomes confounded. When B4 observes that skeleton k' failed in some training problem, that failure could be due to (a) a mode-conflict (color or grasp blocked), or (b) a size-width incompatibility — and B4 cannot distinguish these because its canonical key does not encode the tags. SPECTRE, by contrast, reads the tag atoms from s₀ via Φ_s (per `SPECTRE_METHOD_SPEC.md` §4.1.3) and can in principle learn to disentangle the two failure causes when scoring remaining candidates.

The U/N intuition (suggested by the principal investigator) is the right informal framing: U = unique canonical skeleton signatures encountered in training; N = total skeleton appearances. v0's effective U/N is ~0.002 (≈24 keys / 15 000 appearances). v1 with per-problem tags raises the _effective_ signature space (signature × tag pattern) into a regime where B4's pair-table is undertrained. See §7.3 for quantification.

### 1.3 Required properties of the substrate

For SPECTRE to have a measurable, defensible adaptivity premium over both static (B3) and adaptive (B4) baselines, the substrate must instantiate five properties jointly:

1. **Non-trivial per-scene latent z.** A per-episode random variable such that P(success | s, z) varies substantially with z.
2. **Skeleton-feature non-sufficiency.** z must not be recoverable from the canonical skeleton key alone, so a static prior is forced to marginalize over z.
3. **Failure-observation informativeness.** z must be inferable from within-episode failures, so an adaptive method can condition on ℱ_t and recover.
4. **No universally-safe skeleton.** Structurally guaranteed (cf. v0's P-S1). Required so SPECTRE's adaptivity mechanism actually activates.
5. **Per-problem heterogeneity beyond canonical-key resolution.** True per-skeleton success rate must vary across training problems by an amount that exceeds B4's per-pair-cell estimation noise. This is the v1-specific property; it is delivered by mechanism B.

### 1.4 Design principles

**P1. Discrete-but-product-structured latent modes.** Two axes (color, grasp) of small cardinality (3 × 2 = 6). Discrete modes keep the family structure clean; the product structure forces the latent to be larger than what fits comfortably in a 500-episode pair table.

**P2. Mode-conditional skeleton families via structural guarantee.** Six families F_{ij,g} for color pair {c_i, c_j} and grasp g ∈ {top, side}. Membership determined by the lifted operator sequence and typed-local-id arguments — fully visible to B4. No reliance on cost-bound tuning.

**P3. Per-problem tag confounding.** Width and size tags appear in s₀ but not in operator effects, preconditions, or canonical keys. B4 sees the same canonical keys across all problems; SPECTRE sees the per-problem tags via Φ_s.

**P4. Realistic geometric/physical latent.** Color blockage = "doorway too cluttered while carrying"; grasp mode = "vacuum gripper not seating today, use side grasp"; width/size = "object footprint vs door clearance." All four are recognizable robotics latents.

**P5. Mode prior that punishes marginal greedy.** Non-uniform π over the 6 modes. The default joint prior is the product of the v0 color prior π_color = (0.50, 0.30, 0.20) and the grasp prior π_grasp = (0.60 top, 0.40 side). Tunable.

**P6. Multiple structurally-similar variants per family.** |F_{ij,g}| ≥ 5 after capping, so B3 wastes attempts cycling through within-family variants.

**P7. Open-ended skeleton instance space.** Per-problem tag sampling produces a heavy long-tail of (canonical-key × tag-pattern) effective signatures, so train and test are largely disjoint at the effective-signature level.

**P8. Bilevel-planning compatible but not kinematically simulated.** STRIPS-typed operators for symbolic planning; refinement is a toy multi-gate function, not a geometric sampler. SesameModels shape preserved.

---

## 2. Environment overview

### 2.1 Setting

A 2D mobile-manipulation environment in which a robot transports N items from per-item source zones to per-item target zones across a workspace partitioned into 6 zones connected by 9 passages. The zone graph is the complete bipartite graph K₃,₃ with a fixed matching-based 3-coloring (passages partition into three color classes; each color class is a perfect matching).

All items' source and target zones lie on the same bipartite side; every loaded transport trip therefore traverses two passages of two distinct colors (this is structural, see §4.1). The robot has two grasp modes (top and side). Each problem has per-passage width tags and per-item size tags, sampled at problem-generation time and visible in s₀.

Per-episode latent: a (blocked_color, blocked_grasp) pair specifying which passage color is impassable when loaded and which pick/place grasp mode does not work. Symbolic preconditions encode connectivity, item placement, and the typed structure of operators; they do not encode color blockage, grasp blockage, or size-width compatibility — these are checked only at refinement time. Static atoms in s₀ encode the color of each passage, the width of each passage, and the size of each item.

Conceptual story: a warehouse with shelves on opposite walls, an aisle between them, three cross-aisle corridors of varying widths, and items of varying sizes. On any given day, one corridor is too cluttered to carry a package through, and only one of the robot's two grippers is calibrated. Width-vs-size and color-vs-mode failure causes are independent and confounded at the skeleton level.

### 2.2 Object types

```
Robot           # always exactly 1
Item            # N per problem; default N ∈ {3, 4}
Zone            # always 6, partitioned: L = {L₁, L₂, L₃}, R = {R₁, R₂, R₃}
Passage         # 9 per problem (parent type)
  PassageColorA # subtype: 3 instances per problem
  PassageColorB # subtype: 3 instances per problem
  PassageColorC # subtype: 3 instances per problem
WidthLevel      # fixed: {narrow, medium, wide}
SizeLevel       # fixed: {small, medium, large}
GraspMode       # fixed: {top, side}  (used as static-atom argument; see §2.3)
```

The passage subtyping is load-bearing: typed-local-id renumbering (per `SPECTRE_METHOD_SPEC.md` §4.1.4) groups objects by type, so the canonical key for `TraverseLoadedColorA(robot:0, passage_color_a:0, ...)` is distinct from `TraverseLoadedColorB(...)`. This is what gives B4 access to the color-family signal — the v1 design intentionally does **not** hide color from B4's canonical key. The B4-defeating mechanism is the per-problem tags (mechanism B), not denial of color information.

### 2.3 Predicates

**Dynamic predicates** (change over the course of a plan; appear in operator effects):

```
At(Robot, Zone)
ItemAt(Item, Zone)
HandEmpty(Robot)
Holding(Robot, Item)
HeldGraspTop(Robot, Item)         # set by PickItemTop, required by PlaceItemTop
HeldGraspSide(Robot, Item)        # set by PickItemSide, required by PlaceItemSide
```

**Static predicates** (in s₀ only; never in operator effects):

```
Connects(Passage, Zone, Zone)     # symmetric: emitted in both orderings
PassageWidth(Passage, WidthLevel) # exactly one per passage per problem
ItemSize(Item, SizeLevel)         # exactly one per item per problem
```

The static atoms `PassageWidth` and `ItemSize` are the v1-specific addition. They appear in every abstract state along the skeleton path (because the state abstractor preserves static atoms; see §2.5). They do not appear in any operator's preconditions, add-effects, or delete-effects — so the symbolic planner does not condition on them, and the canonical key for any skeleton is _independent_ of the tag values.

### 2.4 Lifted operators

Eight lifted operators total. Pick and place are split by grasp mode; loaded traversal is split by passage color; empty traversal is generic (any passage subtype).

```
PickItemTop(?robot - Robot, ?item - Item, ?zone - Zone):
  preconditions: { At(?robot, ?zone), ItemAt(?item, ?zone), HandEmpty(?robot) }
  add_effects:   { Holding(?robot, ?item), HeldGraspTop(?robot, ?item) }
  delete_effects:{ ItemAt(?item, ?zone), HandEmpty(?robot) }

PickItemSide(?robot - Robot, ?item - Item, ?zone - Zone):
  preconditions: { At(?robot, ?zone), ItemAt(?item, ?zone), HandEmpty(?robot) }
  add_effects:   { Holding(?robot, ?item), HeldGraspSide(?robot, ?item) }
  delete_effects:{ ItemAt(?item, ?zone), HandEmpty(?robot) }

PlaceItemTop(?robot - Robot, ?item - Item, ?zone - Zone):
  preconditions: { At(?robot, ?zone), Holding(?robot, ?item), HeldGraspTop(?robot, ?item) }
  add_effects:   { ItemAt(?item, ?zone), HandEmpty(?robot) }
  delete_effects:{ Holding(?robot, ?item), HeldGraspTop(?robot, ?item) }

PlaceItemSide(?robot - Robot, ?item - Item, ?zone - Zone):
  preconditions: { At(?robot, ?zone), Holding(?robot, ?item), HeldGraspSide(?robot, ?item) }
  add_effects:   { ItemAt(?item, ?zone), HandEmpty(?robot) }
  delete_effects:{ Holding(?robot, ?item), HeldGraspSide(?robot, ?item) }

TraverseEmpty(?robot - Robot, ?passage - Passage, ?src - Zone, ?dst - Zone):
  preconditions: { At(?robot, ?src), Connects(?passage, ?src, ?dst), HandEmpty(?robot) }
  add_effects:   { At(?robot, ?dst) }
  delete_effects:{ At(?robot, ?src) }

TraverseLoadedColorA(?robot - Robot, ?passage - PassageColorA, ?src - Zone, ?dst - Zone, ?item - Item):
  preconditions: { At(?robot, ?src), Connects(?passage, ?src, ?dst), Holding(?robot, ?item) }
  add_effects:   { At(?robot, ?dst) }
  delete_effects:{ At(?robot, ?src) }

TraverseLoadedColorB(?robot - Robot, ?passage - PassageColorB, ?src - Zone, ?dst - Zone, ?item - Item):
  (preconditions, effects analogous)

TraverseLoadedColorC(?robot - Robot, ?passage - PassageColorC, ?src - Zone, ?dst - Zone, ?item - Item):
  (preconditions, effects analogous)
```

Note that `TraverseLoaded` does _not_ condition on the held grasp (no `HeldGraspTop`/`HeldGraspSide` precondition). The grasp mode affects only the pick/place semantics — once the item is held, traversal is grasp-agnostic at the symbolic level. This keeps the operator count manageable (8 instead of 14) and matches the physical story (the gripper holds the item; the chassis moves).

### 2.5 State abstractor and goal deriver

**State abstractor.** Given a concrete state x, emit a `RelationalAbstractState` whose atoms include:

- Dynamic atoms:
    - `At(robot, zone)` for the robot's current zone
    - `HandEmpty(robot)` if the robot is not holding anything; else `Holding(robot, item)` and `HeldGraspTop(robot, item)` or `HeldGraspSide(robot, item)`
    - `ItemAt(item, zone)` for every unheld item
- Static atoms (these appear unchanged in every state along a skeleton):
    - `Connects(passage, zone_a, zone_b)` and `Connects(passage, zone_b, zone_a)` for every passage
    - `PassageWidth(passage, w)` for every passage
    - `ItemSize(item, s)` for every item

The Objects set contains the robot, all items, all zones, all passages (typed by their color subtype), and the WidthLevel and SizeLevel constants used in static atoms.

The presence of `PassageWidth` and `ItemSize` atoms in s₀ is the access surface SPECTRE's Φ_s uses to read tag information. B4's canonical key (operator-sequence-based) does not include these atoms.

**Goal deriver.** Emit `RelationalAbstractGoal({ItemAt(item_i, target_zone_i) for i ∈ 1..N})`.

---

## 3. Latent scene structure

### 3.1 Scene latent z

```python
@dataclass(frozen=True)
class SceneLatent:
    blocked_color: str   # one of {"A", "B", "C"}, identifying the impassable color class
    blocked_grasp: str   # one of {"top", "side"}, identifying the unusable grasp mode
```

Refinement-time gates (formalized in §5.2):

- `TraverseLoadedColor⟨X⟩` fails when `X == blocked_color`.
- `PickItemTop` fails when `blocked_grasp == "top"`.
- `PickItemSide` fails when `blocked_grasp == "side"`.
- (Place ops are not gated by `blocked_grasp` — see §5.2 design rationale.)

### 3.2 Mode prior π(z)

The default mode prior factors as π(z) = π_color(blocked_color) × π_grasp(blocked_grasp), with:

|Color axis|Prior||Grasp axis|Prior|
|---|---|---|---|---|
|c_A|0.50||top|0.60|
|c_B|0.30||side|0.40|
|c_C|0.20||||

The 6 joint modes have probabilities:

|Mode (blocked_color, blocked_grasp)|π(z)|
|---|---|
|(A, top)|0.30|
|(B, top)|0.18|
|(C, top)|0.12|
|(A, side)|0.20|
|(B, side)|0.12|
|(C, side)|0.08|

⚠ The product-structured prior is a default. A fully-joint 6-cell prior is equally valid and may be tuned in M5–M6. Independence is the simpler null hypothesis; non-independence (e.g., specific (color, grasp) combinations co-occurring) would add another type of confounding signal that SPECTRE could exploit beyond what B4 with axis-marginal modeling could capture. Defer this enrichment unless B4 is still too strong empirically.

### 3.3 Per-problem static tags (mechanism B)

Per problem instance, sample:

- Per-passage width: for each of the 9 passages, sample `width ∈ {narrow, medium, wide}` from a per-passage distribution.
- Per-item size: for each of the N items, sample `size ∈ {small, medium, large}` from a per-item distribution.

The default tag distributions are:

```
P(width)  = (narrow: 0.20, medium: 0.40, wide: 0.40)
P(size)   = (small: 0.30, medium: 0.40, large: 0.30)
```

Compatibility (size ≤ width under the total order small < medium < large vs narrow < medium < wide):

|size \ width|narrow|medium|wide|
|---|---|---|---|
|small|✓|✓|✓|
|medium|✗|✓|✓|
|large|✗|✗|✓|

Under the default distributions, the marginal probability that a random (size, width) pair is compatible is approximately 0.65 — meaning each `TraverseLoaded` op has a ~35% prior probability of being size-incompatible in a randomly-sampled problem. This is the "noise channel" that confounds B4's per-pair conditional estimates.

⚠ The default distributions are tunable. The constraint they must satisfy is **tractability**: the rejection-sampling rate for problems with no feasible skeleton (across all 6 families) must remain manageable (target: ≤ 30% of raw samples rejected). See §4.3 for the rejection procedure.

### 3.4 Skeleton families

Define the _family_ of a skeleton s as the pair `(loaded_color_pair(s), grasp(s))` where:

- `loaded_color_pair(s)` is the unordered pair of distinct colors used by s's `TraverseLoaded⟨X⟩` operators (always exactly 2 colors out of 3, by §4.1's structural property).
- `grasp(s) ∈ {top, side}` is the (single) grasp mode used by s's pick/place operators (always exactly 1 of the 2; mixed-grasp skeletons are pruned, see §5.1).

There are exactly 6 families:

|Family|Loaded colors|Grasp|Succeeds in mode|Marginal success rate|
|---|---|---|---|---|
|F_{BC,top}|{B, C}|top|(A, side)|0.20|
|F_{BC,side}|{B, C}|side|(A, top)|0.30|
|F_{AC,top}|{A, C}|top|(B, side)|0.12|
|F_{AC,side}|{A, C}|side|(B, top)|0.18|
|F_{AB,top}|{A, B}|top|(C, side)|0.08|
|F_{AB,side}|{A, B}|side|(C, top)|0.12|

(Marginal under the default prior, ignoring per-problem tag effects.)

Each family is anti-correlated with exactly one of the 6 modes and incompatible with the other 5. This is the _latent-side_ family structure; the per-problem tag effects (§3.3) modulate per-family per-problem success rates further.

### 3.5 Why per-problem tags break B4 — detailed mechanism

B4 maintains, for each canonical skeleton key k:

- Marginal: p̂(k) ≈ E_problem[ E_z[ 1{s succeeds | s has key k, problem, z} ] ]
- Pair conditional: p̂(k | k' failed) ≈ P_problem,z[ s succeeds | s has key k, k' has key k' attempted in same problem, k' failed ]

**Without per-problem tags** (v0): the only failure cause is mode-conflict. p̂(k | k' failed) cleanly captures mode-conditional posterior; B4 = SPECTRE.

**With per-problem tags** (v1): each failure observation has two possible causes:

- Mode-conflict: blocked_color is in s's loaded color pair, or blocked_grasp matches s's grasp.
- Tag-incompatibility: at least one of s's loaded traversals carries a too-large item through a too-narrow passage.

When B4 observes "k' failed" in some training problem, it cannot tell which cause applied — so its update on "p(k succeeds | k' failed)" is an average over the two cases. In tag-incompatibility cases, k''s failure is _uninformative_ about the mode (the failure cause was orthogonal to z), so the correct posterior on z is unchanged. B4's average-over-causes updates the posterior anyway, polluting the family-inference signal.

SPECTRE's Φ_s reads the static atoms `PassageWidth` and `ItemSize` from s₀. In principle, the learned encoder can compute a per-skeleton "tag-feasibility" feature (for each TraverseLoaded op, is the item-passage pair size-compatible?) and combine this with the failure context to disentangle mode-conflict from tag-incompatibility failures. The empirical question (to be measured in M5–M8) is whether SPECTRE recovers a substantial fraction of this disentanglement signal from 500 training problems; the theoretical upper bound is full recovery.

The closed-form prediction for the B4-vs-SPECTRE gap is therefore not a clean number — it depends on how well SPECTRE actually learns the tag-feasibility feature. §7 provides Monte Carlo expected ranges; the headline is that the gap should be 0.5–1.5 attempts at N=3 under default settings, growing with N (because tag-incompatibility-rate per skeleton scales with skeleton length).

---

## 4. Problem instance generator

### 4.1 Graph topology — K₃,₃ with matching-based 3-edge-coloring (unchanged from v0)

The zone-passage graph is K₃,₃:

```
Zones: L = {L₁, L₂, L₃},  R = {R₁, R₂, R₃}
Passages (9 total), partitioned by color:
   Color A:  L₁-R₁,  L₂-R₂,  L₃-R₃     (identity matching)
   Color B:  L₁-R₂,  L₂-R₃,  L₃-R₁     (left-shift matching)
   Color C:  L₁-R₃,  L₂-R₁,  L₃-R₂     (right-shift matching)
```

Each color class is a perfect matching; the topology is fixed across all instances of the same variant.

Structural properties (proofs by inspection):

(S1) Removing any one color leaves a connected 6-cycle (every problem solvable under every mode-color).

(S2) Every single-color subgraph is a disconnected matching (no single-color same-side path exists, so every loaded trip must use ≥ 2 colors).

(S3) Every same-side zone pair has exactly 3 length-2 paths, one per color-pair: {A,B}, {A,C}, {B,C}.

These are the same v0 properties; v1 inherits them unchanged. They are what guarantee no universally-safe-color skeleton exists.

### 4.2 Sizing table for variants

|Variant|N items|K_pool (raw)|K_pool (capped)|Skeleton length|Modes|Use|
|---|---|---|---|---|---|---|
|RT2D-n3-v1 (default)|3|36|30 (5/family)|13–16 ops|6|Main benchmark|
|RT2D-n4-v1|4|144|30 (5/family)|17–20 ops|6|Transfer-size test|
|RT2D-n2-v1|2|12|12 (uncapped)|9–11 ops|6|Debug / smoke tests|

Raw K_pool = 3 (color pairs) × 2 (grasps) × N! (orderings). Capped to 30 via family-balanced sampling: keep the first ⌈30/6⌉ = 5 skeletons per family in canonical lex order. The cap is uniform across families to preserve the structural-guarantee properties.

### 4.3 Instance sampling procedure

```
Sample one problem instance at N items:

1. Fix the K₃,₃ topology with the matching-based 3-coloring (§4.1).

2. Sample which_side ∈ {L, R} uniformly. All items' sources and targets lie on this side.

3. Sample robot_home ∈ (all 6 zones) uniformly.

4. For each item i ∈ 1..N:
     source_zone_i ∈ which_side-zones uniformly, subject to:
       (a) no two items share a source zone.
     target_zone_i ∈ which_side-zones \ {source_zone_i} uniformly, subject to:
       (b) no two items share a target zone.

5. Sample per-passage width tags: for each of the 9 passages, draw width ~ P(width)
   from §3.3 default distribution.

6. Sample per-item size tags: for each item, draw size ~ P(size).

7. Sample scene latent z = (blocked_color, blocked_grasp) ~ π_color × π_grasp.

8. Tractability check (rejection sampling):
     enumerate the 36 candidate skeletons (closed-form, §5.1)
     compute their family memberships
     for each family F, check whether at least one skeleton in F has all
       its TraverseLoaded ops size-compatible (i.e., size(item) ≤ width(passage)
       for every loaded traversal)
     if no feasible skeleton exists in F_z (the family corresponding to mode z):
       reject this instance and goto step 5 (resample tags)
     if more than (max_rejections=20) consecutive rejections:
       reject this instance and goto step 7 (resample z to find a mode whose
       family is well-served by the current tags)
     if more than max_rejections at step 7 too:
       reject this instance entirely and goto step 5 (resample tags from scratch)

9. Assemble ProblemInstance(
       which_side, robot_home,
       {item_i: (source_i, target_i) for i in 1..N},
       passage_widths, item_sizes,
       scene_latent=z).
```

The rejection-sampling structure ensures every emitted problem has at least one fully-feasible skeleton in the candidate pool (so SPECTRE / B4 / B3 / random can in principle solve it within the budget). The rejection rate at default tag distributions is approximately 10–20% per instance (estimate); ⚠ verify in M3.

⚠ Constraint (a)+(b) at N=4 with 3 zones per side is infeasible (pigeonhole). For RT2D-n4-v1 either (i) relax (a) to allow shared sources, (ii) relax (b) to allow shared targets, or (iii) use a slightly larger graph (K₃,₄). Default: option (i), which adds a "double-pickup" dimension to skeleton diversity. See §9.4.

### 4.4 Train/val/test splits

Per `SPECTRE_METHOD_SPEC.md` §1.5 and §7.2:

- Training: 500 problems, seeds 0–499
- Validation: 100 problems, seeds 1000–1099
- Test: 100 problems, seeds 2000–2099

Problems across splits are sampled i.i.d. from the same generator. The latent z and tags are sampled independently per problem; the resulting splits have approximately the §3.2 mode composition and §3.3 tag distribution in expectation.

For the transfer experiment: train on `RT2D-n3-v1`, evaluate on `RT2D-n4-v1`. The lifted-operator vocabulary is identical across variants; ground-skeleton pool sizes change. Tag distributions and the mode prior are held constant.

---

## 5. Bilevel planning interface

### 5.1 High-level planner: closed-form enumeration

The planner enumerates exactly the 36 (raw) candidate skeletons per problem in closed form, **independent of the per-problem tags** (the planner does not condition on `PassageWidth` or `ItemSize`). The structure of each skeleton is determined by three choices:

- An ordering π ∈ S_N of the items (N! choices).
- A color pair p ∈ {{A,B}, {A,C}, {B,C}} (3 choices).
- A grasp mode g ∈ {top, side} (2 choices).

Given (π, p, g) and the problem's robot_home, source/target zones, the skeleton is constructed deterministically:

```python
def build_skeleton(robot_home, item_order, color_pair, grasp_mode, problem):
    """
    Build the skeleton for a given (ordering, color_pair, grasp) choice.
    All TraverseLoaded ops use passages from color_pair (via the unique
    same-side detour for each item, per K₃,₃ property S3).
    Empty-traversal ops use any color-pair-allowed passage.
    Pick/place ops use grasp_mode.
    """
    ops = []
    current_zone = robot_home
    for item in item_order:
        # Empty-traverse from current_zone to item.source_zone using color_pair
        ops.extend(empty_path(current_zone, item.source_zone, color_pair, problem.topology))
        # Pick
        ops.append(make_pick_op(grasp_mode, item, item.source_zone))
        # Loaded-traverse from source to target via the unique color-pair detour
        loaded_path = loaded_detour(item.source_zone, item.target_zone, color_pair, problem.topology, item)
        ops.extend(loaded_path)
        # Place
        ops.append(make_place_op(grasp_mode, item, item.target_zone))
        current_zone = item.target_zone
    return Skeleton(initial_state=problem.initial_abstract_state, ground_operators=ops)


def enumerate_skeletons(problem):
    skeletons = []
    for item_order in permutations(problem.items):
        for color_pair in [frozenset({"A","B"}), frozenset({"A","C"}), frozenset({"B","C"})]:
            for grasp_mode in ["top", "side"]:
                skel = build_skeleton(problem.robot_home, item_order, color_pair, grasp_mode, problem)
                skeletons.append(skel)
    # Family-balanced cap to K=30 if N >= 3.
    return cap_pool(skeletons, target_size=30)


def cap_pool(skeletons, target_size):
    if len(skeletons) <= target_size:
        return skeletons
    by_family = group_by_family(skeletons)
    per_family = target_size // len(by_family)
    return sum(([s for s in fam[:per_family]] for fam in by_family.values()), [])
```

The closed-form enumerator returns _all_ 36 skeletons regardless of whether they are tag-feasible. Tag-infeasibility is checked at refinement time only. This is essential to mechanism B: B4 sees a uniform pool structure across problems, so its canonical-key accounting is consistent, but per-problem true success rates vary because of tag-incompatibilities the planner did not filter.

**Why no BFS search.** As in v0, the topology is small and fully symmetric, so closed-form is sufficient. A general-purpose BFS enumerator with a 2-color-pure filter and a single-grasp filter remains a valid validation tool — it should produce the same pool modulo enumeration order (test #11 in §8.3).

**Skeleton object structure.** Per `SPECTRE_METHOD_SPEC.md` §4.1.2: a sequence of `(GroundOperator, RelationalAbstractState)` pairs, plus the initial `RelationalAbstractState` s₀. The static atoms `PassageWidth` and `ItemSize` are present in s₀ and propagate through every intermediate abstract state (since no operator deletes them).

### 5.2 Low-level refiner: three-gate latent + tag model

The refiner is a pure function of the skeleton, the scene latent, and the per-problem tags. There is no continuous-parameter sampling and no kinematic simulation.

```python
@dataclass(frozen=True)
class RefineOutcome:
    success: bool
    stuck_at_op_index: int | None
    stuck_cause: str | None       # "blocked_color" | "blocked_grasp" | "size_width" | "noise" | None
    wall_clock: float

def refine(skeleton: Skeleton,
           scene_latent: SceneLatent,
           passage_widths: dict[Passage, str],
           item_sizes: dict[Item, str],
           rng: np.random.Generator,
           base_op_fail_rate: float = 0.02) -> RefineOutcome:
    """
    Three-gate refinement, in order of check:
      Gate 1 (blocked_color):  TraverseLoaded⟨X⟩ fails iff X == scene_latent.blocked_color
      Gate 2 (size_width):     TraverseLoaded⟨X⟩ fails iff size(item) > width(passage)
      Gate 3 (blocked_grasp):  PickItemTop fails iff blocked_grasp == "top"
                              PickItemSide fails iff blocked_grasp == "side"
      Plus residual noise (probability base_op_fail_rate per op).
    """
    SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}
    WIDTH_ORDER = {"narrow": 0, "medium": 1, "wide": 2}

    for i, ground_op in enumerate(skeleton.ground_operators):
        op_name = ground_op.name

        # Gate 1: blocked color (loaded traversals only)
        if op_name.startswith("TraverseLoadedColor"):
            color = op_name[-1]  # "A", "B", or "C"
            if color == scene_latent.blocked_color:
                return RefineOutcome(False, i, "blocked_color",
                                     _sample_fail_time(i, len(skeleton), rng))
            # Gate 2: size-width compatibility
            passage = ground_op.arguments[1]
            item    = ground_op.arguments[4]
            if SIZE_ORDER[item_sizes[item]] > WIDTH_ORDER[passage_widths[passage]]:
                return RefineOutcome(False, i, "size_width",
                                     _sample_fail_time(i, len(skeleton), rng))

        # Gate 3: blocked grasp (pick ops only; place ops not gated)
        if op_name == "PickItemTop" and scene_latent.blocked_grasp == "top":
            return RefineOutcome(False, i, "blocked_grasp",
                                 _sample_fail_time(i, len(skeleton), rng))
        if op_name == "PickItemSide" and scene_latent.blocked_grasp == "side":
            return RefineOutcome(False, i, "blocked_grasp",
                                 _sample_fail_time(i, len(skeleton), rng))

        # Residual noise
        if rng.random() < base_op_fail_rate:
            return RefineOutcome(False, i, "noise",
                                 _sample_fail_time(i, len(skeleton), rng))

    return RefineOutcome(True, None, None, _sample_success_time(len(skeleton), rng))
```

Wall-clock sampling, as in v0:

- `_sample_fail_time(i, L, rng)`: Gamma(α=1.0 + 0.3·i, β=1.0).
- `_sample_success_time(L, rng)`: Gamma(α=1.0 + 0.3·L, β=1.0).

Design rationale for not gating place: gating only at pick is sufficient (any all-top skeleton fails immediately at first PickItemTop in mode (·, top); any all-side skeleton fails immediately at first PickItemSide in mode (·, side)). Gating place too would only add failure modes for skeletons that already passed pick, which doesn't happen for pure-grasp skeletons. This keeps the refinement structure clean and one-fault-per-skeleton per gate.

⚠ The `base_op_fail_rate = 0.02` and the tag distributions in §3.3 are the two main empirical knobs. Their joint effect determines (a) the fraction of failures that are "informative" about the latent (mode-driven failures), (b) the fraction that are "uninformative" (size-width-driven failures), and (c) the fraction that are pure noise. Target ratio: 60% mode / 30% tag / 10% noise across all training failures. Tune in M5–M6.

### 5.3 SesameModels shape

Same `create_bilevel_planning_models` pattern as v0 and as the existing kinder envs:

```python
def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_items: int,
) -> SesameModels:
    ...
    return SesameModels(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        transition_fn=transition_fn,        # symbolic only
        types=types,                         # includes Width/Size/GraspMode types
        predicates=predicates,               # includes PassageWidth, ItemSize
        observation_to_state=observation_to_state,
        state_abstractor=state_abstractor,   # emits PassageWidth, ItemSize atoms
        goal_deriver=goal_deriver,
        skills=skills,                        # 8 lifted skills, one per operator
    )
```

As in v0, `action_space` and `transition_fn` are thin (no continuous control). The continuous-parameter `LiftedParameterizedController` for each skill has a trivial `params_space` and a no-op `step()` — refinement operates at the skeleton level, not the per-step level.

### 5.4 Candidate pool construction (entry point)

```python
def generate_candidate_pool(problem: ProblemInstance) -> list[Skeleton]:
    return enumerate_skeletons(problem)
```

K_pool = 30 for N=3 and N=4 (capped); 12 for N=2 (uncapped). Per-problem pool size is constant (does not vary with tags); per-skeleton feasibility varies.

---

## 6. Data synthesis for SPECTRE training

### 6.1 Per-problem label generation

For each problem instance, attempt every skeleton in the candidate pool exactly once under the problem's scene latent and tags:

```
for problem in dataset:
    S = generate_candidate_pool(problem)
    outcomes = [refine(s, problem.scene_latent,
                       problem.passage_widths, problem.item_sizes, rng)
                for s in S]
    record = ProblemRecord(...)
    save(record)
```

### 6.2 Episode record schema

```python
@dataclass(frozen=True)
class ProblemRecord:
    problem_id: str
    variant: str
    num_items: int
    scene_latent: SceneLatent
    passage_widths: dict[str, str]      # passage_id -> width_level
    item_sizes: dict[str, str]          # item_id -> size_level
    candidate_skeletons: list[Skeleton]
    refinement_outcomes: list[RefineOutcome]

    @property
    def success_indices(self) -> list[int]:
        return [i for i, o in enumerate(self.refinement_outcomes) if o.success]
    @property
    def fail_indices(self) -> list[int]:
        return [i for i, o in enumerate(self.refinement_outcomes) if not o.success]
```

Storage: one JSON-serialized record per problem, in a flat directory:

```
data/RoutedTransport2D/
  n3-v1/
    train/  (500 records)
    val/    (100 records)
    test/   (100 records)
  n4-v1/
    ...
```

### 6.3 F-subset sampling for training examples

Per `SPECTRE_METHOD_SPEC.md` §5.2, training examples are (R, SUCC_R, F) triples. Implementation sketch (subject to confirmation against the training pipeline spec):

```python
def sample_training_examples(record, num_f_samples=8, rng=None):
    fail_set = set(record.fail_indices)
    succ_set = set(record.success_indices)
    all_idx = set(range(len(record.candidate_skeletons)))
    examples = []
    for _ in range(num_f_samples):
        f_size = rng.integers(0, len(fail_set))  # uniform over {0, ..., |FAIL|-1}
        F = set(rng.choice(list(fail_set), size=f_size, replace=False))
        R = all_idx - F
        SUCC_R = succ_set & R
        if not SUCC_R:
            continue
        examples.append(TrainingExample(
            problem_id=record.problem_id,
            candidate_skeletons=record.candidate_skeletons,
            F_indices=F, R_indices=R, SUCC_R_indices=SUCC_R,
        ))
    return examples
```

### 6.4 Compatibility hooks with the training pipeline

```python
# routedtransport2d_data.py — module-level entry points
def make_problem(seed: int, variant: str) -> ProblemInstance: ...
def generate_candidate_pool(problem: ProblemInstance) -> list[Skeleton]: ...
def label_problem(problem: ProblemInstance) -> ProblemRecord: ...
def load_split(variant: str, split: str) -> list[ProblemRecord]: ...
def iter_training_examples(records, **kwargs) -> Iterator[TrainingExample]: ...
```

Contract requirements (unchanged from v0):

1. `Skeleton` objects conform to `SPECTRE_METHOD_SPEC.md` §4.1.2 shape.
2. Predicates and lifted operators in skeletons are present in `SesameModels.predicates`/`.operators`.
3. `RefineOutcome.wall_clock` populated.
4. All object types in skeleton arguments are in `SesameModels.types`.

If the training pipeline spec prescribes a different schema, it wins.

### 6.5 Evaluation driver

The per-episode loop follows `SPECTRE_METHOD_SPEC.md` §6.1 unchanged. As in v0, `record.refinement_outcomes` is pre-computed at data-generation time and indexed by skeleton index; the eval loop does not re-run the refiner, ensuring deterministic comparisons across methods.

---

## 7. Expected statistics and diagnostics

### 7.1 Pool composition

For N=3, K_pool = 30 after family-balanced capping (5 skeletons per family × 6 families). For N=4, K_pool = 30 again (5 per family × 6, but with 24 raw skeletons per family available for sampling; under the cap, 19 per family are dropped). For N=2, K_pool = 12 (uncapped, 2 per family).

These pool sizes are exact and tag-independent; assert in unit tests (§8.3).

### 7.2 Predicted performance — Monte Carlo estimates

Unlike v0, v1's expected statistics are not closed-form because of the per-problem tag heterogeneity. The numbers below are Monte Carlo estimates from the reference implementation, to be verified during M3–M4. Default settings: π_color = (0.50, 0.30, 0.20), π_grasp = (0.60, 0.40); tag distributions per §3.3; base_op_fail_rate = 0.02.

For **N=3** (K_pool = 30, 5 skeletons per family):

|Method|Mean attempts|Notes|
|---|---|---|
|Random ordering|~10–11|Geometric expectation under uniform-random selection|
|B3 (historical marginal)|~5.5–6.5|Cycles families F_{BC,side} → F_{BC,top} → F_{AC,side} → ...|
|B4 (NB log-odds)|~3.0–4.0|Adaptive but partially confounded by tag failures|
|SPECTRE (target)|~1.7–2.2|Disentangles mode from tag failures via Φ_s|
|Mode-oracle|~1.5|Picks the right family first; still subject to tag failures within family|

Predicted **B4 → SPECTRE gap: 1.0–2.0 attempts**. The width of this range reflects empirical uncertainty about how well SPECTRE actually learns the tag-feasibility feature from 500 training problems.

For **N=4**: gap should grow because skeleton lengths are larger (more TraverseLoaded ops per skeleton → more tag-failure opportunities → more confounding for B4). Predicted gap: 1.5–3.0 attempts. ⚠ Verify in M8.

### 7.3 The U/N effective-recurrence metric

Using the metric proposed informally during the v1 design:

- U_canonical = number of unique canonical skeleton signatures encountered in training pools.
- N_canonical = total skeleton appearances in training (Σ |pool_i| over training problems).
- U_effective = number of unique (canonical signature, tag pattern) pairs.

For RT2D-n3-v1 with 500 training problems × 30 skeletons each = 15,000 skeleton appearances:

- Canonical universe: ≤ 36 (closed-form upper bound from (color_pair × grasp × ordering) for N=3). After family-balanced capping, ≤ 30. **U_canonical ≈ 30**, U_canonical / N_canonical ≈ 0.002.
- Effective universe: per-skeleton, the relevant tag pattern is (sizes of the items in this skeleton, widths of the passages used in this skeleton). For N=3, a skeleton uses 3 items and ~4 distinct passages, with each tag drawn from 3 levels. The tag pattern has ≈ 3³ × 3⁴ = 2,187 possibilities, but only some are co-occurring with each canonical key; the realized effective signature space is empirically ~5,000–8,000. **U_effective / N_canonical ≈ 0.4**, into the regime where memorization is severely degraded.

The B4 baseline operates on canonical signatures only — its effective sample size per (k, k') pair is N_canonical / U_canonical² ≈ 17 observations per pair. SPECTRE operates effectively on (signature × tag pattern) — its per-effective-pair sample size is much smaller but compensated by parameter sharing across the continuous embedding space.

This is the formal expression of the "U/N close to 1 → memorization useless" intuition.

### 7.4 Diagnostic metrics

Beyond headline mean-attempts, report:

- **Per-mode success-at-step-t** for each method.
- **Family-inference accuracy (SPECTRE)**: argmax of internal posterior over families given ℱ_t. Should converge to true mode within 1–2 observations on tag-clean problems; degrade gracefully on tag-confounded problems.
- **Failure-cause breakdown**: of all observed failures, what fraction are blocked_color / blocked_grasp / size_width / noise. Target distribution at default settings: ~40% / ~25% / ~25% / ~10%. If size_width fraction < 20% or > 40%, retune tag distributions per §9.2.
- **Tag-feasibility-conditioned premium**: SPECTRE's mean attempts on tag-clean problems (no infeasible TraverseLoaded under any mode) vs tag-confounded problems. Gap should be modest if Φ_s is generalizing well.
- **Wall-clock adaptivity premium**: Σ wall_clock for failed attempts + success wall_clock. Methods that fail fast on early-stuck skeletons (first pick op fails) have a different wall-clock profile than methods that fail late (last traversal fails). Reported alongside attempt count.

---

## 8. Implementation notes

### 8.1 Recommended file layout

```
routedtransport2d/
  __init__.py
  env.py                    # SesameModels factory, state_abstractor, goal_deriver
  operators.py              # The 8 lifted operators, predicates, types
  topology.py               # K₃,₃ topology + 3-coloring
  tags.py                   # WidthLevel, SizeLevel, compatibility table, tag samplers
  problem_generator.py      # make_problem, instance sampling with tractability rejection
  refiner.py                # refine, RefineOutcome, three-gate model
  planner.py                # enumerate_skeletons (closed form) + cap_pool
  data.py                   # ProblemRecord, label_problem, load_split
  training_hooks.py         # iter_training_examples, F-subset sampling
  tests/
    test_topology.py
    test_operators.py
    test_planner.py
    test_refiner.py
    test_families.py
    test_tags.py
    test_tractability.py
    test_end_to_end.py
```

### 8.2 Dependencies

- `relational_structs`, `bilevel_planning`: as in v0.
- `numpy` for RNG.
- `pyperplan`: not required (closed-form planner suffices).

No new dependencies beyond what v0 required.

### 8.3 Validation tests

All assertions are exact unless otherwise marked.

1. **Topology sanity**: K₃,₃, 6 zones, 9 passages, bipartite (no L-L or R-R edges).
2. **Coloring sanity**: each color class is a perfect matching of 3 edges; removing any color leaves a connected 6-cycle.
3. **Operator vocabulary**: exactly 8 lifted operators with the names and signatures of §2.4.
4. **Operator effects consistency**: PickItemTop then PlaceItemTop sequence restores HandEmpty; same for Side. Mixed-grasp Pick/Place sequence is _not_ applicable (placeholder for the planner to filter).
5. **Pool size**: for each variant, K_pool matches §7.1 exactly.
6. **Family partition**: every skeleton in the pool belongs to exactly one of the 6 families; family sizes are uniform after capping (5/5/5/5/5/5 for N=3 and N=4; 2/2/2/2/2/2 for N=2).
7. **No universally-safe skeleton**: for every skeleton, there exists at least one mode (blocked_color, blocked_grasp) under which a gate would fire (color or grasp), independent of tag feasibility.
8. **Tag atoms in s₀**: every problem's initial abstract state contains exactly 9 PassageWidth atoms (one per passage) and N ItemSize atoms (one per item). These atoms persist unchanged in every state along every skeleton.
9. **Refiner three-gate**: synthesize a test skeleton in family F_{ij,g} and check refinement under all 6 modes × all 27 (size, width) combinations; outcomes match the gate logic exactly.
10. **Tractability**: across 500 generated training problems, ≥ 99% have at least one feasible skeleton in their candidate pool. Rejection rate during instance generation ≤ 30%.
11. **Closed-form vs BFS agreement**: for 20 sample problems, the closed-form enumerator and a general BFS enumerator (with 2-color-pure and single-grasp filters) produce the same pool modulo ordering.
12. **End-to-end smoke test**: generate 10 problems, label, run one episode per problem with uniform-random selector, verify no crashes and verify ≥ 60% succeed within budget.

### 8.4 Known simplifications (vs fully geometric TAMP)

- Single-item holding only.
- No item-on-item obstruction at pick or place.
- No irreversibility; any pick can be undone by place.
- Deterministic refinement modulo `base_op_fail_rate`.
- No cost differentiation across operators; all cost 1.
- Tags are fixed per problem (no in-episode tag dynamics).

---

## 9. Open decisions (⚠ — resolve during M3–M6)

### 9.1 Mode prior factorization

Default is independent: π(z) = π_color × π_grasp. A non-independent joint prior could, e.g., make (A, top) and (C, side) co-occur disproportionately, which would add a third axis of confounding signal that SPECTRE could exploit beyond what B4 with axis-marginal modeling captures. Defer enrichment unless B4 is still close to SPECTRE empirically; revisit in M5.

### 9.2 Tag distributions

Default (§3.3): P(width) = (0.20, 0.40, 0.40), P(size) = (0.30, 0.40, 0.30). These are tunable; the constraints are:

- Tractability: rejection rate ≤ 30%.
- Failure-cause balance: size_width fraction in 20–40% of failures.
- Per-family feasibility: each family has ≥ 1 feasible skeleton in ≥ 70% of problems (so all 6 families are exercised in training).

Sweep in M4.

### 9.3 Mixed-grasp skeletons

The current design prunes mixed-grasp skeletons (skeletons that pick item 1 with top and item 2 with side) at the planner level. Mixed-grasp skeletons always fail under the v1 latent (one of {top, side} is always blocked, so any mixed-grasp skeleton fails at the first pick using the blocked mode). Pruning is correct but conservative; if a richer story is desired (e.g., grasp-mode is per-item-feasibility rather than global), the design would change substantially. Defer.

### 9.4 N=4 pool / shared-source vs extended topology

Three options:

1. Allow shared sources (relax instance-generator constraint (a)) — keeps K₃,₃ topology, adds a "double-pickup" skeleton dimension. Default for v1.
2. Allow shared targets (relax (b)) — symmetric to (1) but at place-side.
3. Extend to K₃,₄ — non-trivial because no perfect 3-edge-coloring exists; requires a near-matching argument.

Recommendation: default to (1); revisit only if M8 reveals shared-source-specific failure modes.

### 9.5 Variable-N within a single split

Mixing N=3 and N=4 problems in a single training split is permitted by the SesameModels contract but would interact with the canonical-key universe (different N → disjoint canonical keys). For the headline experiment, hold N fixed per split and use cross-N as the transfer test. Variable-N as a single-split design is a richer-benchmark extension; defer.

### 9.6 Per-passage-and-item width/size correlation

The default tag samplers are independent per passage and per item. A correlated design (e.g., per-zone "shelf height" affecting both items born in that zone and passages adjacent to it) would inject additional structure SPECTRE could exploit via its Φ_s atom-pool. Defer; only enable if the M5 results show SPECTRE close to B4 ceiling on the IID-tag default.

### 9.7 Base op-fail rate

`base_op_fail_rate = 0.02` default. Sweep in M5; target the largest value for which SPECTRE retains > 80% of its ideal premium.

---

## 10. Key terminology

**Anti-correlated families.** Partition F_{BC,top}, F_{BC,side}, F_{AC,top}, F_{AC,side}, F_{AB,top}, F_{AB,side} of the candidate pool, each succeeding in exactly one of the 6 modes and failing in the other 5. Guaranteed by the K₃,₃-plus-same-side topology and the grasp-mode operator split.

**Bipartite partition.** Partition of zones into L-side {L₁, L₂, L₃} and R-side {R₁, R₂, R₃}; all passages cross the partition.

**Canonical key.** B4's per-skeleton signature (lifted-op-sequence with typed-local-id arguments). In v1, the key encodes color and grasp via operator-name and passage-type respectively, but does not encode tag values.

**Color pair / loaded color pair.** The unordered pair of two distinct passage colors used by a skeleton's TraverseLoaded operators. One of {{A,B}, {A,C}, {B,C}}.

**Effective signature.** Pair (canonical key, tag pattern) under which a skeleton's true success rate is determined. Used for U/N analysis (§7.3); not used by any baseline or by SPECTRE directly.

**Empty traversal.** TraverseEmpty op; not gated on the latent or tags.

**Family F_{ij,g}.** Set of skeletons with loaded color pair {c_i, c_j} and grasp mode g. For default N=3, |F_{ij,g}| = 5 after capping (raw 6).

**Grasp axis / blocked_grasp.** Latent axis ∈ {top, side}; blocks PickItemTop or PickItemSide.

**Joint mode prior π(z).** Distribution over the 6 modes; default is π_color × π_grasp.

**K₃,₃.** Complete bipartite graph on 3+3 vertices; admits a 3-edge-coloring into perfect matchings.

**Loaded traversal.** TraverseLoadedColorA/B/C op; gated on (a) blocked_color, (b) size-width compatibility.

**Mode m = (blocked_color, blocked_grasp).** A discrete latent value with 3 × 2 = 6 possible values.

**Per-problem tag.** A static atom in s₀ encoding either a passage's width or an item's size; sampled per-problem at instance-generation time, fixed for the entire episode.

**Problem instance.** (Topology, which_side, robot_home, item-source-target map, passage_widths, item_sizes, scene_latent z). Generated by §4.3.

**Same-side constraint.** Instance-generator rule that all items' sources and targets lie on the same bipartite side.

**Scene latent / z.** Per-episode (blocked_color, blocked_grasp) tuple.

**Size-width compatibility.** Total order: small < medium < large fits narrow < medium < wide. TraverseLoaded with item i through passage p succeeds under gate 2 iff size(i) ≤ width(p).

**Static tag.** Synonym for per-problem tag.

**Variant.** A named environment configuration (e.g., RT2D-n3-v1). Fixes N, mode prior, tag distributions, rejection-sampling parameters.

---

_End of specification. Questions about implementation details that are not resolved here should be resolved by reference to `SPECTRE_METHOD_SPEC.md` (for method-side contracts) or to the training pipeline spec (for data-format contracts), in that order of precedence._