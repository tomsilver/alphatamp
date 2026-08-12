# Porting SPECTRE v3 to a new environment

The generality claim, stated as a checklist. **Two things are required, and neither is a
predicate, a feature, or a fact vocabulary**: a converter (§1) and refiner instrumentation
(§2). The `DomainSpec` (§3) is still read, but since proof-tier demotion was cut from the
method on 2026-07-30 its **axiom declarations are optional** — nothing consumes a proof
unless you opt back in. DD2D is the worked example throughout.

> **Status, updated 2026-08-01.** A transfer has now been attempted: **StickButton2D**,
> via kinder's own env and refiner rather than a bespoke one. It found **two places where
> the contract as written below was wrong**, both now fixed and both recorded in §2b and
> §4:
>
> 1. §2's observation schema assumed the refiner can *name* the objects it failed on
>    (`culprits`). StickButton2D structurally cannot — kinder's collision check returns a
>    bool — so the port needed a second evidence class, not a second fact type.
> 2. §1's table lists `scene_geometry` as required but says nothing about what happens
>    without it: the answer was a training run that exits 0 having written no checkpoint.
>
> The env-2 originally planned here (Khodeir-style 3D sorting on drake-tamp,
> `SPECTRE_v3_proposal.md` §7.6) is still not attempted. Read the cost numbers as
> "measured once, on an environment that shares the substrate", not as fully general.

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
| `scene_geometry` | per-object boundary ring + pose | **required — see §4**; without it the episode is silently skipped |
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

## 2b. If your refiner cannot *name* what it failed on — the second evidence class

The schema above assumes `culprits`: the refiner's validity check knows which objects it
rejected on. **That is a property of the refiner, not a given.** StickButton2D has no such
channel at all — kinder's motion model declines a colliding transition by silently not
moving, and its collision predicate returns a bool without naming anything. A port that
only implemented §2 would emit records with an empty `culprits` list, `coverage` and
`waste` would be identically zero, and v3 would quietly degrade to a static ranker while
reporting no error whatsoever.

The generalisation (`unified_evidence.py`, §2 of the design doc) is that there are **two**
ways a refinement can fail, and an environment may afford either:

| class | what it is | what it emits |
|---|---|---|
| 1 | a validity check rejects the sample **before** a successor state exists | `culprits` — the objects it named |
| 2 | the sample **executes** and the trace check finds observed ≠ predicted | `dev_added` / `dev_deleted` — the deviating atoms |

DD2D is entirely class 1; StickButton2D is entirely class 2. Emit whichever your refiner
affords, in `refiner_metadata["failures"]`; **both keys are always read, and an empty
channel is inert by construction**, so no consumer branches on the environment. Class 2
costs nothing extra to produce if your sampler already compares the achieved abstract state
against the planned one to decide accept-or-reject — which any exact-acceptance sampler
does. Ours simply keeps what that comparison threw away
(`envs/stickbutton2d/instrumented_refiner.py`).

Two traps that cost real time here:

- **Serialize deviations as `(predicate, [arg, ...])` name pairs, and rename them in
  `canonicalize_episode`.** They live in a free-form dict, so nothing type-checks them; if
  the *nested* argument names are not remapped alongside `args`/`culprits`, every record's
  tags silently fail to resolve and the whole stream degenerates to "some failure of some
  schema".
- **Blame derived from a class-2 deviation is not a culprit.** It is stored in its own
  `dev_blame` field, because a culprit was named by the environment and this was inferred
  by us; conflating them would let a model trained where the signal is observed be deployed
  where it is inferred with nothing recording the difference.

**You do not emit `state_delta`.** The record's abstract-state field (`--state-delta`,
§6.1's `s_j`) is derived by STRIPS progression over the candidate's own `operator_seq` from
`initial_abstract_state`, both of which the converter above already supplies. It costs a new
environment nothing, and its predicate vocabulary comes from the same `train_vocab.json`
every other component reads.

## 3. A `DomainSpec` — and its axioms are now optional

> **Changed 2026-07-30.** Proof-tier demotion was cut from the deployed method
> (`decisions.md`), so **"learning is the floor" is the configuration, not the fallback**.
> The `DomainSpec` is still read — it derives `manipulated`, `goal_objects` and
> `length_key` from your operator schema, which the loss and the candidate features need —
> but the `axioms` block below only affects (a) which records are held out of the token
> path as proof-tier and (b) the opt-in `apply_demotion=True` path. **You can port with
> `axioms={}`.** On DD2D the offset was worth 0.23 FP and fired on 6% of rollouts; declare
> the axioms and switch it on if your domain's proofs fire more often.

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

## 4. `scene_geometry` — and the way it fails if you forget

Not optional in practice, and its absence is the single most expensive failure mode in this
guide because **nothing reports it**. `train_v3._trainable()` filters every episode without
geometry, so `n_train` reaches 0, `deployed_val_fp` returns `inf`, `improved = inf < inf` is
never true, and the run terminates with **exit code 0 and no `best.pt`** — after however
many hours the collection took.

Build it from whatever your environment already exposes, and prefer *its own* geometry
helper to re-deriving shapes; `envs/stickbutton2d/scene_geometry.py` calls kinder's
`object_to_multibody2d`, the same function its renderer and collision checker use, so the
recorded footprint cannot drift from the one the refiner enforced. Every key of
`object_registry` needs an entry (invariant I5).

Three things that are easy to get silently wrong:

- **Pose convention.** kinder's `Rectangle(x, y, w, h, theta)` takes the **lower-left
  corner**; the schema wants the centroid, and the ring wants to be centred on it. Reading
  one as the other displaced the 1.25-long stick by 0.625 world units, with nothing to
  notice.
- **Multibody objects.** Record the part that is actually a static footprint. The
  StickButton2D robot is base + arm + gripper; the arm is *configuration*, and only the
  base collides with the table, so the base disc is what is stored.
- **The normalization frame.** `dataset_v3` divides poses by a frame width read from
  `SceneGeometry.frame`. Those keys were DD2D literals (`drawer_w`/`drawer_d`); they now
  accept generic `frame_w`/`frame_d` as well. Write one spelling or the other — as of
  2026-08-08 an **absent frame raises** (naming the fix) instead of silently meaning
  `scale = 1.0` / unnormalized coordinates.

**The goal channel needs nothing per-environment (2026-08-08).** v3 reads `obj_is_goal` —
1.0 for any object named by the goal atoms, computed by `spec.goal_objects` — not the old
`obj_is_target`, which presupposed a single distinguished target and was silently all-zero on
any env whose goal names several objects. A new environment supplies **nothing** here; the
boolean is derived from the goal it already declares, and is correct for any number of targets.
`ObjectGeometry.is_target` stays in the schema (stored, unread by the v3 tensorizer). The scene
relation `obj_rel` is likewise the anchor-free triple `[area, sinθ, cosθ]` — the target-anchored
offsets and the privileged `concave` flag are gone, so there is no target-relative block to
degrade. (An inference-time probe priced this removal at **Δ 0.00 FP** on both deployed models;
`notebook/07` 2026-08-08.)

## 5. Porting the *comparator*, not just SPECTRE

Everything above ports the re-ranker. A new environment is only a *result* once the
low-level baseline runs there too — otherwise the representation question has no second
side. `piginet/` is env-agnostic since 2026-08-01; an adapter answers three questions
(`piginet/domain.py`) and nothing else:

| | what it supplies | DD2D | StickButton2D |
|---|---|---|---|
| vocabulary | glosses + a stable word order | 21 words | 17 words |
| numeric scales | `frame_extent`, `shape_max` | cm, 50×40 drawer | m, 3.5×2.5 world |
| data | `problems(split)`, `crops(split, pid)` | JSON tree + PNGs | `EpisodeRecord` + rasteriser |

**The scales are the part that will bite you, and it fails silently.** PIGINet divides
poses by the frame and shapes by per-field maxima so both land in `[-1, 1]`. Ported against
another domain's constants, StickButton2D's shape features read `|mean| 0.0061` (max 0.05)
instead of `0.372` — a channel ~60× flatter, i.e. off. Nothing raises; the baseline just
looks hopeless, and "the low-level predictor loses here" is then a unit bug you are about
to publish. **Print the normalised feature distributions before training anything.**

Two more things the StickButton2D port needed that are not obvious:

- **Synthesise pose literals if your abstract initial state has none.** SB2D's `s_0` is two
  atoms and names no coordinates. A low-level predictor that receives no positions is not
  one, so the adapter emits an `at-pose` literal per object, mirroring what DD2D's records
  carry natively. Without it the comparison is rigged in the abstraction's favour.
- **Build the examples from the same records SPECTRE trains on.** Then the two methods'
  labels are identical *by construction* rather than by agreement, and any gap between them
  is about representation rather than about two separately-produced label sets.

Ask early whether your environment's perception is informative at all. On SB2D every
unpressed button renders identically, so PIGINet's image channel separates only
{button, stick, robot} — which the type literals already give. That does not invalidate the
row, but it bounds what it shows, and it is much better known before the run than after.

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
unrecoverable — demotion is a finite offset, never removal) behaviour. **Since 2026-07-30
the default resolves this by not declaring anything**: demotion is off, so a wrong `local`
cannot cost you anything unless you opt in. Prefer measuring what the offset would buy
before declaring it.
