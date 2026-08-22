# Restock3D v2 — porting map for the four learned methods (SPECTRE, PIGINet, LAZY, VLMPlan)

> **Status: info-gathering map (2026-08-18).** A verified inventory of what each method requires and
> what is missing, produced *before* any collection code is written, so an early mistake does not force
> a full restart. This is a reference, not an implementation plan; the phased build it prepares for is
> in §8. Every file:line reference was taken from the current tree; the load-bearing facts were
> confirmed against on-disk data (§ "Verification").

## Context

We have a working **Restock3D v2** environment (kinematic PyBullet, continuous packing:
`pick` / `place_tall` / `place_short`, two shelf sections, F2 crowding + F3 height + reach-over
feasibility). It is an *env milestone* — env + oracle + Stage-0 gate + demos + a geometry-informed
plan-gen prior — with **all data-collection machinery deliberately deferred**. Before spending the
expensive collect→train→eval cycle on the four methods, this document maps the full gap.

**Two decisions taken (2026-08-18):**
1. **Collection target = Restock3D v2 (continuous packing).** v1 (discrete regions) is superseded and
   used only as the *plumbing template*.
2. **Geometry representation = full 3D point cloud.** Widen the shared `SceneGeometry` schema and
   SPECTRE's encoder to a real 3D point set (rather than a 2D footprint + height scalar). This is the
   most invasive choice and touches all four methods' geometry paths — mapped in detail below.

All paths are under `src/alphatamp/approaches/spectre/` unless noted `experiments/…` or `data/…`.

---

## 0. Executive summary (the shape of the work)

- **One shared blocker dominates everything: there is no Restock3D `SceneGeometry` producer**, and the
  shared schema is **2D-footprint-only**. Confirmed on real collected data that `scene_geometry is None`
  in every `restock3d_v1` episode. All four methods read this layer (SPECTRE tokens, PIGINet
  crops+pose/shape, LAZY node geometry, VLMPlan render+text).
- **The 2D schema is inadequate in principle, not just unimplemented.** Cubes (`small_half =
  (0.025,0.025,0.025)`) and tall blocks (`tall_half = (0.025,0.025,0.12)`) have **identical 2D
  footprints** and differ **only in height** — and height is exactly what F3 turns on (a 0.24 m block
  clears the 0.34 m tall section but not the 0.15 m short section). A 2D-footprint predictor cannot
  tell a cube from a tall block, i.e. it is blind to the environment's headline hardness. → the chosen
  3D point cloud is *required*, not polish.
- **What already works and transfers:** the failure-evidence / culprits channel (verified: real class-1
  culprits in the collected data), pool enumeration over a real `SesameModels`, and the oracle. The v1
  collection *plumbing* exists and is the template for the v2 plumbing we must write.
- **Per-method env-specific surface is small and well-templated:** LAZY = a 5-field dataclass; PIGINet =
  one domain adapter; VLMPlan = one env-adapter + one labeler. But every one is gated behind the
  **shared prerequisites**: a v2 collection that emits 3D `scene_geometry`, a vocab, and (for the
  learned rankers) a trained SPECTRE-style dataset.
- **Critical fair-comparison subtlety:** a *top-down* render shows cube and tall block as identical
  squares, so **PIGINet's image channel and VLMPlan's snapshot must convey height** (oblique/perspective
  camera — the env already has PyBullet cameras — or an explicit side/elevation view / height labels),
  or those two methods are handed a degenerate image on the very axis that matters.

**Dependency order (what unblocks what):**
```
   3D SceneGeometry producer  ─┐
   v2 instrumented refiner ────┼─►  v2 collection pipeline ──►  vocab ──►  SPECTRE train
   v2 collection registration ─┘                                  │            │
                                                                  ├─►  LAZY (reuses episodes+vocab)
                                                                  └─►  PIGINet (native data + crops)
   VLMPlan (adapter+labeler, needs only the env + a labeled render, not the trained dataset)
   compare EnvSpec + cache driver  ── wires all rows into the table
```

---

## 1. The shared blocker — 3D `SceneGeometry` (prerequisite for SPECTRE, PIGINet, LAZY, and the compare render)

### 1.1 What exists / what's missing

- **Schema (2D):** `schema.py:89-124` — `ObjectGeometry(name, pose:(x,y,θ), boundary:2D-ring, family,
  area, concave, is_target)`, `SceneGeometry(objects, containers, units, frame)`.
- **Producers today:** only `envs/stickbutton2d/scene_geometry.py` and the DD2D converter
  (`envs/dd2d/spectre_convert.py:150-191`). `collect.collect_episode` builds `scene_geometry` **only**
  for `stickbutton2d` (`collect.py:495-502`). **Restock3D emits none** (grep-confirmed; verified
  `scene_geometry is None` on `data/spectre/raw/restock3d_v1/train/episodes/ep_0.pkl.gz`).
- **Why it silently breaks SPECTRE:** `train._trainable` (`train.py:227-232`) drops every geometry-less
  episode → a run finishes with **no checkpoint**. `dataset.build_example` also raises without
  `scene_geometry` (`dataset.py:441-442`) and without a `frame_w/frame_d` normalization extent
  (`dataset.py:461-474`). Invariant **I5** (`schema.py:229-235`) requires geometry for *every*
  `object_registry` key once `scene_geometry` is set.

### 1.2 The 3D extension (chosen representation)

The point cloud is **analytic from ground-truth geometry** — objects are axis-aligned cuboids with
known half-extents (`kinematic_env.Restock3DEnvConfig.small_half/tall_half/clutter_half`) and a 3D
pose; no depth camera or perception is involved. A producer samples each object's surface (or its 8
corners / a face grid) in the item frame (centroid at 0) plus the world pose.

Concrete surfaces the 3D choice touches (config-gated so the 2D DD2D/SB2D path is byte-unchanged and
the `test_v3_equivalence` 2D oracle still holds):

- **Schema** (`schema.py`): add optional 3D fields to `ObjectGeometry` (e.g. a `point_cloud:
  tuple[tuple[float,float,float],…]` and a 3D `pose`/height), leaving the 2D `boundary`/`pose` intact
  for existing envs. Extend `SceneGeometry.frame` with a z / depth extent for normalization.
- **SPECTRE encoder** (`encoders.py`): `FootprintEncoder.point_mlp` `nn.Linear(2, D_POINT)` →
  `Linear(3, …)` (`encoders.py:161`); `obj_boundary` `(B,N,P,2)` → `(B,N,P,3)` (`encoders.py:95`);
  `SceneEncoder.pose_proj` `nn.Linear(3, D_POSE)` → widened for the 3D pose (`encoders.py:199`);
  optionally add height/volume to `obj_rel` (currently `[area,sinθ,cosθ]`, `D_REL=3`, `encoders.py:58`).
  A 3D checkpoint is a **new model config** (width-bound), so the flag defaults off.
- **SPECTRE dataset** (`dataset.py`): a 3D point sampler replacing/augmenting `resample_ring`
  (`dataset.py:83-105`); read the 3D pose + point cloud + z-normalization in `build_example`
  (`dataset.py:488-498`).
- **Producer (new):** an `envs/restock3d/scene_geometry.py` (template: `envs/stickbutton2d/
  scene_geometry.py`) reading the env's movable bodies + half-extents → `SceneGeometry`, and a
  `collect.py` branch that calls it for the restock model (mirror `collect.py:495-502`).

### 1.3 Baseline geometry channels the 3D choice also touches

- **PIGINet** shape scalars `[w,h,area,concave]` + pose `[x,y,θ]` (`baselines/piginet/encoders.py`
  value channels; `shape_max`/`frame_extent` divisors) → 3D analogs `[w,h,d,volume,…]`, pose
  `[x,y,z,θ]`.
- **LAZY** geometry node features `geom_dim=8 = (x,y,sinθ,cosθ,w,h,area,concave)`
  (`baselines/lazy/graph.py:65-72`) → add z / height / depth; `feasibility.py` untouched.
- Both degrade to zeros if `scene_geometry` is absent, so the producer is the gate.

---

## 2. SPECTRE — full requirement map

**Data path:** on-disk `EpisodeRecord` → `canonicalize_episode` → `build_example` /
`build_record_arrays` → `collate` → `SpectreBatch` → `SpectreModel.forward`.

**The `EpisodeRecord` contract** (`schema.py:179-258`): `initial_abstract_state` (x0-free),
`goal_atoms`, `object_registry` (name→type), `skeleton_pool` (`SkeletonRecord`, `schema.py:74-80`),
`outcomes` (`OutcomeRecord`, `schema.py:163-176`, incl. `refiner_metadata` + `refinement_wall_clock_s`),
`scene_geometry` (§1). **Failure evidence** is derived, never stored: `FailureRecord`
(`failure_record.py:115-235`) with `culprits` (class-1) + `dev_blame`/`state_delta` (class-2);
`coverage`/`waste`/`culprit_pool` in `unified_evidence.py` (`:319-334`, `:479-500`, `:555-587`).

| SPECTRE needs | Status on Restock3D v2 | Source / template |
|---|---|---|
| Pool of candidate skeletons per problem, each refined non-short-circuiting + labeled | **Substrate exists, not wired to collect.** Real `SesameModels` (`models_v2.build_restock3d_v2_models`) + pool generators (`plan_generator_v2.py`, hff). No pipeline refines *every* member for v2 (oracle refines only 1). | v1 collect refines a 200-skeleton pool (verified). `collect.collect_episode:371-433` |
| x0-free abstract s0 + goal atoms + object→type registry | **Works** (abstractor `RestockAbstractorV2`, goal = `Stored(o)`) | `models_v2.py` |
| 3D `scene_geometry` (+ normalization frame) | **MISSING (blocker).** See §1. | new `envs/restock3d/scene_geometry.py` |
| Instrumented failure evidence (`refiner_metadata["failures"]`: step, schema, args, culprits, n_step, exhausted, budget_exhausted, dev_added/deleted/blame) | **Exists for v1 operators only.** Payload verified on disk with real culprits. But the refiner is bound to the **3-arg `place`** op (`instrumented_refiner.py:278,295`), not v2 `place_tall`/`place_short`. Needs a **v2 recording sampler** + continuous-section-capacity F2 attribution. | `envs/restock3d/instrumented_refiner.py` (v1) → port to v2 ops. `failure_metadata:386-419` |
| `DomainSpec` (env_variant → `QueryAxioms`) | **MISSING for v2** (`domain.py` has only `restock3d_v1 → EMPTY_SPEC`). `EMPTY_SPEC` (all hint-tier) is a legitimate honest start; a proof-tier F3 spec can come later. | `domain.py:56-79,152-220,281` |
| Vocab (operators/predicates/types, train-only) | **Not built** (no `train_vocab.json` for restock). Populated from v2 ops/preds/types once collection exists. | `vocab.py:109-200`; `spectre_build_vocab.py` |
| Collection registration: `model_name` dispatch, `strata_v2` (`ENV_VARIANT="restock3d_v2"`), env config, aug policy | **All MISSING for v2** (v1 versions exist as templates). | `collect.py:48,78,238,273`; `strata.py`; `conf/env/restock3d_v1.yaml`; `env_registry.py:73-77,196` |

**Net for SPECTRE:** the *hard* signal (culprits/evidence, pool) exists in v1 form; the work is (a) the
3D geometry producer + encoder widening, (b) porting the instrumented refiner to v2 ops, (c) the v2
collection registration (a mechanical clone of v1's), then (d) run a real pool collection + vocab +
train. LAZY and PIGINet both depend on (a)-(d) landing first.

---

## 3. PIGINet — env-adapter map

Consumes **per-object crops** (frozen CLIP ViT-B-32, 512-d each; `baselines/piginet/encoders.py:52-101`)
⊕ pose ⊕ shape scalars, over the **`PIGINetDomain` protocol** (`baselines/piginet/domain.py:36-84`).

To run PIGINet on Restock3D v2, provide (template = `baselines/piginet/sb2d_adapter.py`):
1. A **`PIGINetDomain`** impl: `name`, `vocab` (stable order — indexes the frozen CLIP-text cache),
   `gloss(word)`, `frame_extent`, `shape_max`, `problems(split)`, `crops(split, pid)`, **and
   `object_names`** (Protocol-plus, `sb2d_adapter.py:255-259`).
2. A **`GLOSSES`** table for every op/pred/type word.
3. **Numeric scales** (`frame_extent`, `shape_max`) — 3D analogs; config-derived (`_config_scales`).
4. **Examples from the collected v2 episodes**: one `PIGINetExample` per (skeleton, outcome),
   `label=(outcome=="success")`, with **synthesised `at-pose` literals per object** (the only route the
   concrete pose reaches the low-level predictor; `dataset.py:34`, `sb2d_adapter.py:219-221`) — for 3D,
   pose `[x,y,z,θ]` + shape `{w,h,d,volume,concave}`.
5. **Per-object crops** — a scene renderer + per-object crop (template: kinder path
   `sb2d_render_convert.py` + `SB2DKinderDomain.crops`). **Must convey height** (oblique/perspective or
   side view) or the image channel is blind to tall-vs-short.
6. **Dispatch**: a branch in `train.py::_build_domain` (`:399-431`) + `_PIGINET_PATHS` / `cache_piginet`
   in `precompute_dd2d_cache.py` (`:76-229,699-793`).

Depends on: the v2 collection (episodes) + the 3D scene geometry + a height-aware renderer.

---

## 4. LAZY — env-adapter map (cheapest, once SPECTRE data exists)

GAT policy over a prefix-tree of the pool + online feasibility ϕ. **Consumes the same `EpisodeRecord`
pickles + the same `train_vocab.json` as SPECTRE — no images, no native data.**

To run LAZY on Restock3D v2, provide:
1. **A `LazyDomain` + a `make_lazy_domain` prefix branch** (`baselines/lazy/domain.py:21-68`) — just
   `name`, `env_variant`, `frame_extent`, `shape_max`. **This is the only env-specific code.**
2. The v2 **episodes** + **vocab** (produced by SPECTRE's collection).
3. **3D geometry**: the node geometry (`graph.py:65-72`) widens for height (else zeros). No
   culprit-channel dependency — falls back to SB2D-style suffix-blame in `feasibility.py` if absent
   (no code change).
4. **Cache wiring**: train `python -m …lazy.train --env-variant restock3d_v2` → checkpoint at
   `checkpoints/restock3d_v2/lazy_s<seed>/ckpt.pt`; `cache_lazy` (`precompute_dd2d_cache.py:1109-1186`)
   already env-agnostic once the variant validates.

Depends on: SPECTRE's v2 collection + vocab landing first (then LAZY is ~a day).

---

## 5. VLMPlan — env-adapter map (independent of the trained dataset)

Attaches a **full-scene labeled snapshot** + generates plans; off-pool proposals are refined live and
charged. Two protocols (`baselines/vlmplan/adapter.py:47-161`): **`EnvAdapter`** (15 methods) +
**`Labeler`**.

To run VLMPlan on Restock3D v2, provide (template = `sb2d_adapter.py` + `sb2d_label.py`):
1. **`EnvAdapter`** impl: `skills`, `objects`, `type_ancestors`, `controllers_str`,
   `typed_objects_str`, `type_hierarchy_str`, `goal_str`, **`init_state_str` with geometry disclosure**
   (height, section clearances, reach-over — the abstract model hides these), `images`, **`ground`
   against the *full* domain operators** (`_domain_operators`, `sb2d_adapter.py:134-194`), `pool_index`
   (in-pool proposals reuse the stored outcome, never re-refined), `canonical_key`, `plan_str`,
   `published_order`, `discretionary_objects`.
2. **A labeled scene renderer** → `images()` with **Set-of-Mark labels on canonical
   `object_registry` names** (template: `render_kinder_labeled_scene` + `_annotate_scene`,
   `envs/stickbutton2d/render.py:257-315`). **Must convey height** (oblique PyBullet camera / side view)
   or the snapshot shows identical squares. `images() -> []` gives the text-only LLMPlan arm.
3. **A `Labeler`** (template: `sb2d_label.py::SB2DOffPoolLabeler`): live off-pool refinement whose
   sampler/budget/seed rule **exactly matches the v2 collection**; must pass `score.label_agreement`
   (`score.py:469-514`).
4. **Registry branches** in `registry.py::make_adapter` + `make_labeler_factory` (`:21-66`).

Depends on: the env + a v2 pool (for `pool_index`) + a height-aware render. **Does not need the trained
SPECTRE dataset**, so it can proceed in parallel — but its off-pool labeler must mirror the collection's
refiner settings, so it is cleanest *after* the collection config is frozen.

---

## 6. The v2 collection pipeline + comparison glue (shared plumbing)

**v2 collection (clone of the v1 plumbing, retargeted to v2 ops):**
- A `model_name` (e.g. `restock3d_v2`) dispatched in `collect.py` `_make_env_models` /
  `_make_trajectory_sampler` / `_failure_metadata_fn` (`:78,238,273`) → `create_restock3d_v2_models`
  + a v2 recording sampler.
- A `strata_v2` (`ENV_VARIANT="restock3d_v2"`), a `conf/env/restock3d_v2.yaml`, a `domain.py` entry,
  and an `env_registry` aug-policy entry (`region`→False stays; cuboid/robot→True).
- A **pooling + reject-and-resample collector** analog of `experiments/spectre/sb2d_collect.py` (the v1
  `restock3d_collect.py` is only a thin per-stratum loop, no reject-resample). Rejects problems with
  `num_success==0`.
- Budgets: 500 train / 100 val / 100 test; recalibrate `K_max` / per-candidate cap from **real**
  per-candidate refinement (today only a geometric-proxy `kmax_estimate.json` and oracle-only sweep
  timings exist).

**Comparison table** (`compare_envs.py`, `compare.py`, `precompute_dd2d_cache.py`):
- One `EnvSpec` (`compare_envs.py:30-112`, add to `ENVS` `:517`): `key`, `title`,
  `env_variant="restock3d_v2"`, `stratum_labels`, `stratum_meaning`, a **`render_scene`** + a
  **`plan_label`**, flags. `stratum_of` (`compare.py:124`) assumes seed→stratum banding — v2 problem
  ids must match that convention (as SB2D deliberately did) or supply their own recovery.
- Cache-driver onboarding: `_PIGINET_PATHS` entry, `_REFINE_CAP_S` entry, `_configure_paths`
  validation. Method registration (`STATIC_METHODS`, `SPECTRE_FAMILIES`, `SEQUENCE_METHODS`,
  `LAZY_FAMILIES`) is already env-agnostic.

---

## 7. What already works and is reused (de-risking)

- **Failure evidence / culprits** — verified on disk: `refiner_metadata["failures"]` with real class-1
  culprits (`cube_goal3` blamed for blocking `cube_goal2`). F1/F2/F3/reach-over probes exist
  (`grasp_blockers:85`, `reach_over_culprits:134`, `_probe_place:292`, `_probe_pick:346`). (v1-op-bound
  → port to v2 ops.)
- **Pool enumeration** over a real `SesameModels` (`models_v2.build_restock3d_v2_models`) — hff +
  geometry-guided generators already enumerate distinct goal-reaching skeletons.
- **Oracle** certifies r0–r3; `refine_skeleton_v2` is a working (manual) refiner.
- **Baseline cores are env-agnostic** — only the per-env adapter/dataclass/registry-branch is new.
- **Canonicalization already remaps** restock-style `refiner_metadata` object names
  (`canonicalize._remap_refiner_metadata:182-246`).

---

## 8. Recommended build order (phased; each phase gated before the next spends compute)

1. **3D `SceneGeometry` producer** (`envs/restock3d/scene_geometry.py`) + schema/encoder widening +
   a unit test that a built episode has geometry for every registry object (I5). *Gate: a 1-episode v2
   collection has non-None 3D `scene_geometry`.*
2. **v2 instrumented refiner** (recording sampler on `place_tall`/`place_short`, continuous-section F2
   attribution) — reuse the v1 grasp/reach corridor code. *Gate: F3/F2/reach-over emit correct culprits
   on a known scene.*
3. **v2 collection registration** (model_name dispatch, strata_v2, env config, domain spec, aug policy)
   + a reject-resample collector. *Gate: a 5-problem collection writes valid `EpisodeRecord`s; label
   agreement / oracle sanity holds.*
4. **Small pilot collection** (e.g. 20/stratum) → **vocab** → **SPECTRE train (1 seed)**. *Gate: a
   checkpoint is produced (proves geometry wired) and beats the naive planner order — the packing
   negative-control expectation is that the abstract advantage is bounded here.*
5. **Full collection (500/100/100)** → **SPECTRE 3-seed**, then **LAZY** (episodes+vocab reused),
   **PIGINet** (native crops), **VLMPlan** (adapter+labeler), then the **`EnvSpec` + compare cache**.

---

## 9. Risks / gotchas (each has cost real time before)

- **Geometry-less "successful" training** — a run finishes with no checkpoint because every episode was
  dropped (`train.py:227-232`). The Phase-1 gate exists to catch this.
- **Top-down render blindness** — PIGINet's image and VLMPlan's snapshot must convey height, or those
  two methods are tested on a degenerate image on the F3 axis (fair-comparison flaw).
- **3D checkpoint is width-bound** — keep the 3D encoder config-gated so DD2D/SB2D 2D checkpoints and
  the `test_v3_equivalence` oracle are unaffected.
- **`canonicalize_episode` is not idempotent** — tensorize from raw episodes only.
- **Stride, never truncate**; **DD2D-style generation is `PYTHONHASHSEED`-dependent** (restock uses
  seeded generation — confirm reproducibility of the v2 collection).
- **v2 refiner-to-op mismatch** — the existing refiner silently no-ops on v2 skeletons (it branches on
  `a.name=="place"`); the port must rebind to `place_tall`/`place_short`.
- **`refine_skeleton_v2` is a manual rollout, not the real `BacktrackingRefiner`** — the collection
  should use the real refiner path (via `collect.py`) for label parity with the other envs, not the
  oracle's manual certifier.

---

## Verification of this map

- Every file:line reference was gathered from the current tree (three parallel exploration passes +
  direct reads of `schema.py`, `encoders.py`, the `kinematic_env` config, and a real `restock3d_v1`
  episode).
- Load-bearing facts confirmed by inspecting on-disk data:
  - `data/spectre/raw/restock3d_v1/train/episodes/ep_0.pkl.gz` → `scene_geometry is None`;
    `refiner_metadata["failures"]` present with a real culprit (`cube_goal3` blamed for blocking
    `cube_goal2`); pool of 200 skeletons, 125 success / 75 fail. (This data is *stale* — old
    `mujoco_tidybot_robot` types, discrete-region v1 — but it proves the plumbing path and the geometry
    gap.)
  - `Restock3DEnvConfig`: `small_half=(0.025,0.025,0.025)`, `tall_half=(0.025,0.025,0.12)`,
    `section_clearances=(0.34,0.15)` → cube and tall block share a footprint and differ only in height.
