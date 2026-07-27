# Porting SPECTRE v3 to a new environment

The generality claim, stated as a checklist. **Three things are required, and none of them
is a predicate, a feature, or a fact vocabulary.** DD2D is the worked example throughout;
its entire environment-specific content is reproduced below and is eight lines.

> **Status.** This is the *architectural* generality claim — the contract is small and the
> fallback path is measured. It has **not** yet been demonstrated by an actual transfer:
> env-2 (Khodeir-style 3D sorting on drake-tamp, `SPECTRE_v3_proposal.md` §7.6) was not
> attempted. Read it as "here is the interface and here is what it costs", not as
> "transfer is verified".

---

## 1. A converter: your episodes → `EpisodeRecord`

Everything downstream — vocab, dataset, training, evaluation — consumes only serialized
`EpisodeRecord` pickles. The substrate (`SesameModels`, gym, refiners) exists so a
*collector* can generate them; if you already have data, you never touch it.

One `EpisodeRecord` = one problem, and must carry:

| field | meaning | notes |
|---|---|---|
| `skeleton_pool` | the candidate plans | `operator_seq` of ground operators |
| `outcomes` | per-candidate `success` / `fail` | plus `refiner_metadata` (below) |
| `scene_geometry` | per-object boundary ring + pose | the model is geometry-aware; without it the episode is skipped |
| `goal_atoms` | goal literals | `goal_objects` is derived from these |
| `provenance` | `env_variant`, `problem_id` | `env_variant` selects the `DomainSpec` |

`envs/dd2d/spectre_convert.py` is a complete worked converter (JSON → `EpisodeRecord`).

## 2. Refiner instrumentation: emit a `FailureObservation` where a query fails

This is the only change to *your* planner, and it is deliberately the cheapest possible
one: **record what the refiner already computed.** At each site where a continuous query
fails, emit

```python
FailureObservation(
    step_index=j,           # which plan step failed
    schema="pick",          # the query/stream schema — your own name for it
    args=(obj,),            # the objects the query was about
    culprits=(blocker,),    # objects the failed samples collided with
    unmoved=frozenset(...), # objects the prefix has not moved (derivable from the state)
    n_step=n, exhausted=..., budget_exhausted=...,
)
```

**Observation-only is a hard invariant, not a style note.** In DD2D `n_attempts` *is* the
stream call counter, so one extra call shifts it and cascades into every label. The fix
there was to reuse the witness the collision short-circuit had already computed
(`grasp_cfree` → `grasp_blocker(...) < 0`) rather than run a new check. Verify differentially
against a pre-instrumentation collection — DD2D's came out identical on 290/290 candidates.

`culprits` is what makes evidence useful and is worth the plumbing: it is the *observed*
counterpart of a "what is blocking this" predicate, and being observed is exactly what makes
it legal (C2) where computing it ourselves would not be (L2's `clears`).

If you cannot emit observations at all, the system still runs — records are backfilled from
`failure_action`-style metadata, more weakly. See `failure_record.py`.

## 3. A `DomainSpec`: per-query axioms, and nothing else

```python
_DD2D = DomainSpec(
    axioms={
        "retrieve":     QueryAxioms(monotone=True, local=True, exact=True),
        "pick":         QueryAxioms(),
        "place-buffer": QueryAxioms(),
    },
    min_calls_per_schema={"pick": 1, "place-buffer": 2, "retrieve": 1},
)
```

That is the whole of DD2D's environment-specific content. Two questions per query type:

- **`monotone`** — does failing against an occupancy set imply failing against any superset?
  (Universal in collision-based feasibility; usually yes.)
- **`local`** — do the objects the prefix moved actually leave the query's relevant region?
  (A *world-layout* property. True in DD2D because staged objects go to a separate buffer;
  **false** in, e.g., same-surface decluttering. When in doubt, do not declare it.)
- **`exact`** — is a completed run of this query exhaustive? Separate from whether it
  *ran*, which the observation reports. Getting this split wrong is what made v2.2 demote
  12 genuinely-feasible plans on a budget-exhausted candidate.

Declaring an axiom has the epistemic status of writing the PDDL domain file: it is
specification, not inference. **An unknown variant degrades to `EMPTY_SPEC`** (everything
hint-tier, nothing demoted) rather than raising — "learning is the floor" is the default
path, not a special case.

`min_calls_per_schema` is only used to derive the conservative exactness witness for
*backfilled* records; with real instrumentation it is unused.

---

## What you do **not** provide

Worth stating explicitly, because the previous version of this system needed all of them:

- no per-environment predicate (`clears`, `blocked-at-contents`, …);
- no fact-type vocabulary — the query schemas come from your domain file;
- no geometry routine at inference — the refiner reports, we do not re-derive;
- no per-dataset knobs — the short-first prior was removed (R1) precisely because it helped
  one collection and diverged training on another.

`manipulated`, `goal_objects` and `length_key` are all derived from the operator schema and
the goal literals; on DD2D the derived versions were verified identical to the hand-written
ones on 120000/120000 skeletons.

## Cost, honestly

For DD2D the instrumentation was ~4 emission sites plus one refactor to avoid an extra
stream call, and the spec is 8 lines. The re-collection it forced was ~1.6 h. The converter
is the largest piece and is entirely about *your* data format.

The part that does not transfer for free is the **`local` axiom**: it is a claim about world
layout, and it is the one place where a wrong declaration produces wrong (though never
unrecoverable — demotion is a finite offset, never removal) behaviour. Prefer not declaring
it and measuring what you lose.
