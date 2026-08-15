# Restock3D — Oracle Solver (design & build guide, v0.1)

A privileged solver for Restock3D that constructs a correct skeleton directly from instance
metadata and refines it through the standard refiner. Its **primary deliverable is empirical
budget calibration** — the per-candidate refinement cap and per-episode budget should be set
from measured oracle solve times, not guessed — with three secondary roles it gets almost for
free: the P5 oracle row, per-instance feasibility certification, and an audit of how close the
observable eager heuristic comes to privileged knowledge.

**Epistemic tags** as in the sibling docs: **[E]** established, **[D]** derived by construction,
**[P]** registered prediction (probe named), **[?]** unverified.

---

## 1. Purpose and role boundaries

### 1.1 Why budgets need an oracle

The per-candidate refinement-abandonment cap is a deployment knob with real teeth: on DD2D/SB2D
it was what made SPECTRE-adaptive fastest-to-first-success, by cutting exactly the expensive
near-feasible traps [E, as_built §10.5]. On Restock3D nobody knows what the cap should be —
attempts run up to ~2 minutes and may still fail [E, observed]. The cap cannot be set from
*infeasible*-candidate times (those are unbounded by definition; the cap's job is to truncate
them). It must be set from the distribution of **time to refine a candidate that is actually
feasible** — and the only way to sample that distribution without first solving the collection
problem is a solver that reliably produces feasible skeletons. That is the oracle.

The one hard correctness constraint this induces [D]: **the cap must upper-bound feasible
refinement time at high quantile.** A cap below some instances' feasible-refinement time
truncates feasible candidates → they are labeled failed → label corruption that no amount of
training fixes. `budget_exhausted` exists precisely to keep truncation distinct from true
exhaustion (`proves_failure() = exhausted ∧ ¬budget_exhausted` [E]), but labels of *feasible*
candidates must not depend on that distinction firing.

### 1.2 What the oracle feeds — and what it must never feed

The earlier concern stands: oracle plans are geometry-aware and typically longer (relocations,
deliberate ordering), so they are distributionally unlike enumerator output. That is a problem
**only** if oracle plans enter places reserved for the enumerator. The boundary:

| consumer | oracle allowed? | why |
|---|---|---|
| per-candidate cap / episode budget | **yes — primary role** | budgets need feasible-refinement times; distributional shape of the plan is irrelevant to a wall-clock quantile |
| P5 baseline↔oracle gap row (oracle FP per stratum) | **yes** | this row is *defined* as a privileged reference |
| instance certification (feasible? `d = (k, σ)` label correct?) | **yes** | a certifier is supposed to be privileged |
| eager-heuristic development | audit only (§6, OV4) | via the certificate-diff, logged; never by tuning weights to match oracle plans |
| **skeleton pools** (membership) | **no** | pools come from one enumeration config per dataset — the standing discipline; if pools lack feasibles, fix the enumerator or the instance, don't inject an alien plan |
| **training signal / labels** beyond feasibility certification | **no** | labels are refinement outcomes of pool members, full stop |

With that boundary the "looks different" objection dissolves: nothing that cares about plan
shape ever sees an oracle plan [D].

---

## 2. Design overview — invert the generator

The instance generator forward-samples a configuration it knows to be feasible: it places talls
against tall slots, computes σ from the slot arithmetic, authors blocking adjacencies and counts
k [E, proposal §4]. The oracle simply **re-derives a witness plan from the same metadata**
(persisted at generation time per the earlier decision: blocking graph, slot table, clearances,
`d = (k, σ)` all ride the instance record). Three consequences:

- *Why it works is by construction, not hope* [D]: every generated instance is feasible under
  the generator's own model; the oracle searches a space the generator already certified
  non-empty. The residual gap — sampler margins, marginal blocking angles, MuJoCo/pybullet slop
  — is covered by a bounded repair loop (§3.4), and its size is itself a diagnostic.
- *Measurement validity*: the oracle refines through the **standard refiner with the standard
  samplers** — no sampler hints, no privileged poses. Its wall-clock is therefore the same
  quantity the methods pay, which is what makes the budget transferable [D]. (Corollary: any
  refiner change — e.g. the cheap-checks-first reordering — invalidates the calibration; one
  refiner version per calibration, same invariant as one per dataset [E].)
- *One certificate library, three consumers*: the `fits` / `slots` / blocking logic already
  exists twice — in the generator (source of truth) and in the eager heuristic's observable
  tables. The oracle is the third consumer; factor the library so all three share code, and the
  observable-vs-privileged table diff becomes a free audit (§6).

## 3. Components

### 3.1 O1 — privileged tables

`restock3d/oracle/tables.py`: load blocking graph, per-region slot counts and clearances, and
object dims from instance metadata. Assert agreement with the generator's recorded `d = (k, σ)`
(a corrupted or stale metadata record should fail loudly here, not surface as a mystery later).

### 3.2 O2 — assignment solver

Solve object→slot assignment exactly. At Restock3D scale (N ≤ ~8 objects, ≤ 6 regions, ≤ ~16
slots) plain backtracking or brute-force matching is sufficient — milliseconds; no ILP
dependency warranted [D]. Constraints: talls only to tall-region slots; per-region load ≤
`slots[R]`. Deterministic under a fixed seed (the standing `PYTHONHASHSEED` caveat applies to
anything upstream of generation [E]). Prefer assignments maximizing residual tall-slot slack
(cheap tie-break that buys refinement robustness).

### 3.3 O3 — sequencer

Order the assigned steps:

1. **Relocations first**: topologically order clutter moves over the blocking graph (blocker
   moved before any pick it blocks; chains handled by the topo order; a cycle is a generator
   bug — assert). Each relocation is `Pick(c); PlaceBuffer(c, buffer)` using the ground buffer
   regions (the §3 prerequisite of the heuristic guide — the oracle needs it too).
2. **Per-region FFD**: within each region, largest footprint first, talls before smalls — the
   crowding-robust order [D, standard bin-packing].
3. Region order: deterministic and otherwise arbitrary (e.g., tall regions first).

### 3.4 O4 — refinement harness + repair loop

Refine the skeleton with the standard refiner under instrumented timing (reuse the existing
counter/effort machinery [E] — the oracle's attempts are `FailureRecord`s like anyone else's,
which is also what makes its FP row well-defined). On failure, the oracle is *allowed* to read
the witness and repair:

- **crowding witness** → re-sequence (offender later / elsewhere) or re-assign around the
  contested region;
- **blocking witness not in the graph** → add the edge, re-run O3 (this catches
  marginal-gap blocking the generator's model missed — log it, it is R-b evidence);
- **height witness** → assert-fail: by construction this cannot happen ([D], `H_t ≥ c_s + 0.05`);
  if it does, the metadata or the check is wrong — stop and investigate.

Bounded at R = 5 repair rounds. Exhausting R does **not** mean "hard instance" — it means the
instance escaped the generator's feasibility model and must be routed to certification (§5).

### 3.5 O5 — outputs

Per instance: `{certified_feasible, T_oracle, n_attempts (oracle FP), plan_len, per-phase timing
(BiRRT / IK / checks), repair events, table diff vs eager}`. Per stratum: quantiles of
`T_oracle`, mean oracle FP, rejection rate.

## 4. Budget calibration — the recipe

1. Run the oracle over ~30–50 instances per stratum (r0…r3), 2–3 sampler seeds each [?] (enough
   for stable p95; widen if across-seed spread is large).
2. **Per-candidate cap, per stratum:** `cap_r = p95(T_oracle | success, stratum r) × 1.5`,
   rounded up. Per-stratum rather than global because r3 plans are longer (relocations + more
   objects) and a global cap sized for r3 wastes wall-clock truncating nothing on r0/r1 [D].
   Sanity anchor: the DD2D/SB2D precedent set caps at 2 s / 10 s as deployment knobs [E]; expect
   Restock3D's to be an order of magnitude larger given ~2 min observed attempts.
3. **Episode budget:** `cap_r × K_planned_attempts` plus enumeration time — the number that
   decides whether K = 40–60 full labeling is affordable at all, i.e. the collection go/no-go.
4. **Recalibrate** whenever the refiner changes (cheap-checks-first reordering, sampler edits) —
   the calibration is a property of a refiner version, and the reordering is expected to shrink
   `T_oracle`'s check-dominated component substantially [P, read off the per-phase timing].
5. Record the calibration artifacts (quantile tables, refiner version hash) alongside the
   dataset config, so a collection can state which calibration it ran under.

The per-phase timing decomposition is a deliberate side-product: if BiRRT/IK dominate even on
*feasible* refinements, that quantifies exactly how much the §7.3 refiner reordering can save
and how much is irreducible motion-planning cost [P].

## 5. Instance certification (closing the V2 loop)

The heuristic guide's V2 failure branch says "no feasible in the top 50 is an
instance-generation problem, not a heuristic problem." The oracle is what makes that branch
decidable: for any suspect instance it answers *does a feasible plan exist at all* — separating
**enumerator miss** (oracle succeeds; the informed order or K needs work) from **generator
miss** (oracle exhausts repairs; the instance is outside its own feasibility model).

Certification policy: never silently drop. Log, regenerate with a fresh seed, and track the
**rejection rate per stratum**. Rejection rate ≤ ~2% [?] is sampler-residual noise; a high rate
means σ/k targeting is mis-tuned and the correct move is back to P3, with the rejection log as
the evidence. Certified-feasible becomes a precondition for an instance entering any collection
— the guarantee DD2D-style labeling implicitly relies on.

## 6. The observable-vs-privileged audit (OV4, free bonus)

Diff the eager heuristic's observable tables against the oracle's privileged ones per instance:
blocking-set agreement, slot-count agreement, fits agreement. If the diff is ≈ 0, that is a
reportable sentence: *the observable eager order recovers privileged feasibility knowledge
almost exactly on this domain* — which both strengthens the eager heuristic's story and bounds
what any static feasibility predictor (PIGINet included) could in principle extract from the
initial state [P]. If the diff is material, it localizes exactly which quantity (usually
`slots`, via sampler margins) the observable model gets wrong — actionable either way.

## 7. Implementation steps

1. **Generator metadata persistence** — confirm/complete: blocking graph, slot table,
   clearances, `d`, seed, into the instance record at generation time (already planned; make it
   load-bearing now).
2. **Certificate library factoring** — one module consumed by generator, eager tables, oracle.
3. **O1–O3** — tables loader, backtracking assigner, sequencer (+ the `PlaceBuffer`/ground-
   buffer prerequisite if not yet merged). Pure functions, unit-tested against P1/P2 fixtures.
4. **O4** — refinement harness on the standard refiner with timing instrumentation and the
   repair loop; oracle attempts recorded as ordinary `FailureRecord`s.
5. **O5 + calibration script** — quantile tables per stratum; writes the budget-calibration
   artifact.
6. **Certification hook** in the instance generator pipeline (certify-on-generate, with the
   rejection log).

Rough scale: O1–O3 are an afternoon on top of the certificate library; O4–O6 a day or two of
harness work, dominated by wiring the timing decomposition [?].

## 8. Validation

- **OV1 — solve rate**: oracle certifies ≥ ~98% of generated instances within R = 5 repairs,
  per stratum [P]. *Abort:* materially lower → P3 (generation tuning), with repair logs as the
  diagnosis.
- **OV2 — calibration stability**: `T_oracle` quantiles stable across sampler seeds; caps
  derived and committed.
- **OV3 — oracle FP row**: mean oracle attempts per stratum small (target ≤ ~2 [P]) — this *is*
  the P5 oracle reference against which the astar gap is measured.
- **OV4 — table audit** as §6.

## 9. Risks and invariants

| id | risk / invariant | guard |
|---|---|---|
| O-a | cap below feasible-refinement time → label corruption | §1.1 argument; p95 × 1.5, per stratum; re-check after any refiner change |
| O-b | oracle time inflated by sampler variance, caps oversized | multi-seed quantiles (OV2); report spread |
| O-c | oracle plans leak into pools or training | §1.2 boundary table; enumeration-config-per-dataset discipline |
| O-d | repair loop silently papers over generator bugs | height-witness asserts; repair events logged and reviewed, not just retried |
| O-e | calibration reused across refiner versions | calibration artifact carries the refiner version hash; loader refuses a mismatch |
| O-f | privileged metadata drifts from the live env config | O1 asserts metadata ↔ recorded `d` agreement on load |
