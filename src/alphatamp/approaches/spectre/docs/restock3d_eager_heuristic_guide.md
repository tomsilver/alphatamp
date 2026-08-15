# Restock3D — The Eager-Validity Heuristic (build guide, v0.1)

How to make A\* surface feasible Restock3D skeletons early — without an oracle — by evaluating
the refiner's own validity checks eagerly at the initial state and folding them into
state-dependent action costs. Covers what it is, why it is justified, why it should work, how to
implement it in the KinDER / bilevel-planning stack, and how to validate it before any
collection run depends on it.

**Epistemic tags** as in the Restock3D proposal: **[E]** established (code/docs/operation),
**[D]** derived by construction, **[P]** registered prediction with its probe named,
**[?]** unverified assumption a step below must confirm.

---

## 1. The problem this solves

hff is structurally blind on Restock3D **by design** [D]. Heights, slot capacities, crowding
order, and grasp blocking were deliberately kept below the abstraction line, so under delete
relaxation every pick-and-place sequence to any region looks equally goal-reaching; hff values
collapse to plan length and the enumeration order within a length class is tie-breaking noise.
Consequences observed so far: r0/r1 solve within <6 attempts [E], r2/r3 unknown because each
refinement attempt costs up to ~2 minutes in 3D PyBullet [E], making blind enumeration
unaffordable to even *measure*.

What we need for collection: an ordering such that the **first feasible skeleton appears within
the first ~5–10 refinement attempts**, produced by the **same enumerator over the same grounded
action set** (so feasible plans are ordinary pool members that surface early, not products of an
alien generator with a different length/shape distribution), using **only information available
in the observable concrete/abstract state** (so it is not an oracle and could in principle be
consumed by any method that reads the initial state, e.g. PIGINet).

## 2. What the heuristic is, and its epistemic status

**Eagerly-evaluated validity checks as action costs.** The refiner already owns the geometric
tests it will run during refinement — `grasp_cfree_3d`, height-vs-clearance, region extents.
Evaluate those same tests once at the initial state, cache them as static tables, and charge
penalized action costs during search wherever a step contradicts them. This is the standard
TAMP move (PDDLStream's eager test-stream evaluation is the canonical form) and it reads
nothing beyond the observable initial state: object bounding boxes (`bb_*` features [E]),
region poses/extents (env state [E]), and the output of the model's own validity checks.

Status: **a model component, like the samplers** — specification, not a learned routine and not
an oracle. The one-line justification for the paper: *the informed order consults the planning
model's own validity checks, evaluated eagerly at the initial state; no generator internals, no
refinement outcomes, no privileged solver.*

Two things it is deliberately **not**:

- **Not a change to the shared abstraction.** Adding e.g. a `Fits(o, R)` predicate to the PDDL
  would strengthen *every* method, delete the F3 (height-mismatch) failure family from the
  evidence stream for everyone, and shrink the r2/r3 gap the environment was built to create. If
  a typed-abstraction baseline is ever wanted, it is a separate named arm — a paper decision,
  not a collection decision.
- **Not pruning.** Penalties are large-but-finite. Provably-doomed candidates (tall→short
  placements) must remain *in* pools — they are the F3 evidence source SPECTRE learns from —
  they just must not be refined *first*.

## 3. Prerequisite: a relocation operator (check before building)

The `Pick` penalty below charges for *unmoved* blockers, which presupposes the abstraction can
move clutter at all. The proposal's K3 defined only `Place(robot, obj, region)` onto shelf
strips; r1 solvability requires clutter to go **somewhere**, and "back on the ground" is
indistinguishable from "never moved" if `PlaceGround` merely re-adds `OnGround` [D].

**Required fix (small, uses existing machinery):** author one or more **ground buffer regions**
in the task JSON (`"target": "ground"` + `ranges` — the region language already supports this
[E]) and add `PlaceBuffer(robot, obj, buffer_region)` adding `InRegion(obj, buffer)`. Then
*moved* is a pure abstract-state test: `moved(c, s) ≡ ∃R: InRegion(c, R) ∈ s`. This also keeps
the DD2D analogy exact (`place-buffer` is the discretionary step waste's backward-relevance
pass classifies [E]). If clutter may instead be shelved, that couples relocation with slot
capacity — a difficulty escalator; keep buffers on the ground in v1.

## 4. The three static tables (computed once per problem)

All three are pure functions of the initial concrete state; total cost is milliseconds-to-
seconds per problem, amortized over the entire search.

| table | definition | source | exactness |
|---|---|---|---|
| `fits[o][R]` | `bb_z(o) ≤ clearance(R)` | object bbox + region/cell extents | **exact by construction** [D]: tall blocks were sized `H_t ≥ c_short + 0.05`, so the strict test is robust to any modeling slop; ε = 0 |
| `slots[R]` | `floor(W_strip(R) / (w_obj + margin))` | region width + object footprint + the sampler margin constant | exact up to the single-row slot model — the *same* arithmetic the instance generator uses (§3.6 of the proposal), re-derived from observable geometry rather than generator metadata [D]; mismatch risk is a registered check (V0) |
| `blockers[o]` | `{c : grasp_cfree_3d(o, poses₀) rejects because of c}` | literally call the P1 check at initial poses | as exact as the check itself; the generator authored adjacencies to block all sampled approach angles [E, generator design], so initial-pose evaluation suffices [?, confirmed by V0] |

Also derive, from the goal + tables: `tall_goal = {o : ¬fits[o][R] for every short R}`,
`tall_regions = {R : fits[o][R] for tall o}`.

## 5. The cost function

Keep hff as the distance-to-go estimate; put the information into **state-dependent action
costs** so A\* orders by penalized `g` and top-K enumeration yields least-suspect plans first.
(Honest note: `h` stays unit-cost while `g` is penalized, so `h` is inadmissible w.r.t. the new
objective — irrelevant here, because the objective *is* the enumeration order, not optimality
[D].)

```
cost(a, s) = 1 + penalty(a, s)

Place(o, R):
    λ_h · [¬fits[o][R]]                                   # T1 height: provable dead end
  + λ_c · [load_s(R) ≥ slots[R]]                          # T2 strip overflow
  + λ_r · [o ∉ tall_goal ∧ R ∈ tall_regions
           ∧ tall_free(s) − 1 < tall_demand(s)]           # T3 slot squatting
  + λ_o · load_s(R) · footprint(o) / footprint_max        # T4 crowding risk (soft, optional)

Pick(o):
    λ_b · |{c ∈ blockers[o] : ¬moved(c, s)}|              # T5 grasp still obstructed

PlaceBuffer(o, B):  0                                     # relocations are never penalized
```

State functions (all counts over abstract atoms — cheap): `load_s(R) = |{o : InRegion(o,R) ∈
s}|`; `tall_demand(s) = |{o ∈ tall_goal : Stored(o) ∉ s}|`; `tall_free(s) = Σ_{R ∈ tall_regions}
max(0, slots[R] − load_s(R))`; `moved(c,s)` as in §3.

**Weight hierarchy** — coarse on purpose; do not enter a tuning loop [?]:

| weight | nominal | rationale |
|---|---|---|
| λ_h | 50 | must dominate any plan-length trade (plans are ≤ ~2N+2k ≈ 20 steps), while staying finite so doomed candidates remain enumerable for F3 evidence |
| λ_b | 8 | must exceed the ~2-step cost of a relocation, so a genuinely blocked pick prefers `PlaceBuffer` over eating the penalty |
| λ_c, λ_r | 8 | same tier: violating capacity/reservation should outweigh any ordering convenience |
| λ_o | 1 | tie-break only — implements first-fit-decreasing (largest-footprint-first within a region), the textbook bin-packing rule; drop it if V2 shows it adds noise |

## 6. Why it works — the family-by-family argument

The penalties are the Tier-A certificates of the earlier design discussion, recast as costs and
re-derived from observable state. Each maps onto exactly one designed hardness source, which is
why effectiveness is close to by-construction rather than hoped-for:

| designed hardness | failure family | term | why the term is aligned |
|---|---|---|---|
| tall object under short ceiling | F3 | T1 | infeasibility was *sized into the objects* (`H_t ≥ c_s + 0.05`) precisely so it is decidable from bboxes alone [D] |
| strip capacity binding | F2 | T2 | same slot arithmetic that defines σ in the strata [D] |
| smalls squatting tall slots (the motivating pathology) | F2 | T3 | reservation arithmetic makes a small-into-tall placement expensive exactly when remaining tall demand needs the slot [D] |
| order fragility within a region | F2 | T4 | FFD ordering; standard and defensible |
| authored grasp blocking | F1 | T5 | consults the *identical* check the refiner will run [E] |

Corollaries worth stating in any writeup:

- **r0/r1 no-op property** [P, tested by V1]: on slack, lightly-blocked instances every penalty
  is ≈ 0 and the order reproduces hff's. If it doesn't, something is miswired — this is the free
  regression test.
- **Residual FP is sampler noise** [P]: among zero-penalty candidates the remaining failures are
  continuous-level (unlucky poses), which the strata were designed to keep small relative to the
  abstract-level infeasibility mass.
- **No alien-plan problem** [D]: same grounded actions, same enumerator, different `g` — feasible
  plans (including longer relocation plans on r1/r3) are ordinary enumeration members surfaced
  early. Longer-with-relocations is the true shape of feasible plans on blocked instances; the
  dd2d_v3 short-first-prior incident is the standing reminder that burying long feasibles is the
  failure mode, not a property to preserve [E].

## 7. Implementation

### 7.1 Wiring (order of work)

1. **Tables module** — `restock3d/eager_tables.py`: pure functions
   `build_tables(initial_state, task_spec) → EagerTables` computing §4. Reuse the P1
   `grasp_cfree_3d` implementation and the region extents already exposed for the converter (K2)
   [E]. Persist the table dump alongside collection metadata — it costs nothing and gives
   per-skeleton penalty breakdowns for diagnostics.
2. **Attach at model creation** — compute in `create_bilevel_planning_models(...)` for the
   restock3d env-model and carry on the `SesameModels` bundle so the task planner can see it.
   (Locate the exact hand-off point in the SeSamE call path when wiring; treat the attachment
   site as a to-find, not a known API [?].)
3. **Cost hook in the search** — wherever successor `g` is accumulated in the A\* used for
   skeleton enumeration, replace unit cost with `1 + penalty(a, s)`. The penalty is a pure
   function of `(action, abstract_state, tables)`; determinism is inherited (note the standing
   `PYTHONHASHSEED` caveat for anything upstream of generation [E]).
4. **Config flag** — a named planner arm (e.g. `planner=astar-eager`) so informed/blind orders
   are switchable per run and per method row. Never let the flag default silently.
5. **Logging** — per-skeleton: total penalty, per-term breakdown, rank under both orders. This
   is what later decides whether `astar-eager` becomes a *reported* named arm.

### 7.2 Reference pseudocode

```python
@dataclass(frozen=True)
class EagerTables:
    fits: dict[tuple[str, str], bool]        # (obj, region) -> fits
    slots: dict[str, int]                    # region -> slot count
    blockers: dict[str, frozenset[str]]      # obj -> initial-pose blockers
    tall_goal: frozenset[str]
    tall_regions: frozenset[str]
    footprint: dict[str, float]

def penalty(action, s, T, w) -> float:
    if action.name == "PlaceBuffer":
        return 0.0
    if action.name == "Place":
        o, R = action.obj, action.region
        p = 0.0
        if not T.fits[o, R]:
            p += w.h
        load = sum(1 for x in objs if ("InRegion", x, R) in s)
        if load >= T.slots[R]:
            p += w.c
        if o not in T.tall_goal and R in T.tall_regions:
            demand = sum(1 for x in T.tall_goal if ("Stored", x) not in s)
            free = sum(max(0, T.slots[r] - load_of(r, s)) for r in T.tall_regions)
            if free - 1 < demand:
                p += w.r
        p += w.o * load * T.footprint[o] / max(T.footprint.values())
        return p
    if action.name == "Pick":
        o = action.obj
        unmoved = sum(1 for c in T.blockers[o]
                      if not any(("InRegion", c, r) in s for r in regions))
        return w.b * unmoved
    return 0.0
```

(~60 lines with the table builder; no new geometry code beyond what P1/K2 already produce.)

### 7.3 The refiner-side companion (bigger wall-clock win)

Independent of the ordering, most of the 2 min/attempt is presumably BiRRT + IK burned on
samples a millisecond geometric test would reject [?]. Reorder the refiner to run the cheap
validity checks (the same `fits` / cfree / in-region tests) **before** motion planning, so
abstractly-doomed candidates die in seconds. Two standing invariants apply [E]:

- **One refiner version per dataset.** `n_attempts` *is* `counter.calls`; a mid-collection
  refiner change shifts every label. Reorder first, then collect — and run the differential
  replay check (the 290/290 precedent) to confirm label/steps-bound/failure-action invariance on
  a replayed candidate set.
- **Use the budget machinery, not uncapped refinement.** A 30–60 s per-candidate cap with
  `budget_exhausted` recorded is exactly what the record schema distinguishes from true
  exhaustion (`proves_failure()` stays meaningful).

## 8. Governance — what the informed order may touch

Pool **membership** and baseline **order** are separable, and the decision is consequential
because the r2/r3 gap is the thing the environment exists to exhibit:

| use | verdict |
|---|---|
| collection ordering (find feasibles fast, label pools affordably) | **yes** — the whole point |
| pool coverage (ensure pools contain feasibles at all) | **yes**, with V3 guarding F3 presence |
| the reported classical-baseline row | **no by default** — report plain length/hff order over the *same* pool |
| an additional named arm (`astar-eager`) | optional, decided later from the §7.1(5) logs; beating "astar with the model's own eager checks" is a strictly stronger result than beating blind hff, if the numbers bear it out |

## 9. Validation plan (afternoon-scale, before any collection)

- **V0 — table unit tests.** Against the authored P1/P2 fixture scenes: `blockers` matches the
  scenes' designed blocking sets; `slots` matches what the place sampler actually achieves on a
  packed strip (the slot-model-mismatch check); `fits` matches refiner outcomes on tall→short
  attempts. *Abort:* slot arithmetic and sampler disagree by >1 slot → fix the margin constant
  before proceeding.
- **V1 — r0/r1 regression.** Penalties ≈ 0 everywhere; rank correlation with hff order ≈ 1;
  first-success attempt count unchanged (<6) [P].
- **V2 — first-feasible position on r2/r3.** ~10 instances/stratum, enumerate top-K under the
  informed order, refine in order under the cap. *Success:* first feasible within the top 5–10 →
  collection viable at K ≈ 40–60 with full labeling. *Failure:* even the informed order finds no
  feasible in the top ~50 → **this is not a heuristic problem** — it is a pool-coverage /
  instance-generation problem (σ too tight, blocking unresolvable, or buffer capacity
  insufficient), and the correct move is back to P3 tuning, *before* burning collection compute.
  This ordering of conclusions is the P3-before-P5 gate discipline from the proposal.
- **V3 — F3 preservation.** Top-K pools on r2/r3 still contain tall→short candidates (λ_h finite
  is doing its job); if not, lower λ_h or raise K.
- **V4 — refiner-reorder differential replay** (if §7.3 is adopted): label invariance on a
  replayed set before the reordered refiner produces any dataset.

## 10. Registered risks

| id | risk | guard |
|---|---|---|
| R-a | slot model vs sampler margins disagree → T2/T3 misfire | V0 |
| R-b | `blockers` at initial poses misses angle-dependent partial blocking | V0 fixtures include the marginal-gap scenes from P1 |
| R-c | weight sensitivity invites a tuning loop | weights are declared coarse (§5); only λ_h/K may move, and only via V3 |
| R-d | informed order leaks into a reported baseline row silently | the §7.1(4) named flag + §8 table |
| R-e | penalties change pool *composition* enough to shift the learned-method comparison | pools for all reported methods come from one enumeration config per dataset — same discipline as one-refiner-per-dataset |
