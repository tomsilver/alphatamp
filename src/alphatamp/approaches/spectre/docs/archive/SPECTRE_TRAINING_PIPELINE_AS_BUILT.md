# SPECTRE: Training Pipeline — As Built

**Companion to `SPECTRE_TRAINING_PIPELINE_SPEC.md`. This document describes the
pipeline that actually exists under `src/alphatamp/approaches/spectre/`, and
calls out where it diverges from the spec.**

The spec's three-layer architecture (Layer 1 raw episodes / Layer 2 derived
artifacts / Layer 3 online-sampled training examples) was the design target.
In practice we kept Layer 1 and Layer 3 essentially as specified, but
**collapsed Layer 2** — flat parquet tables, prior caches, and the
manifest-as-parquet are not implemented. EDA reads raw episodes directly.

---

## 1. Module map

| Module | Role | Spec section |
|---|---|---|
| `schema.py` | Frozen dataclasses for `EpisodeRecord`, `ProvenanceBlock`, `SkeletonRecord`, `OutcomeRecord`, `SummaryBlock`. Self-validating via `__post_init__`. | §5 |
| `config.py` | `CollectionConfig` (frozen, hashable, YAML round-trippable). | §4 |
| `collect.py` | `collect_episode`, `collect_and_save`, `collect_and_save_result` (worker-safe). Non-short-circuiting refinement. | §6 |
| `io.py` | `atomic_write_pickle_gz`, `load_episode`, `list_episodes`, `scrub_partial_writes`. | §3.2, §5.8, §6.5 |
| `trajectory.py` | `apply_operator`, `reconstruct_trajectory` — STRIPS progression to recover intermediate states on demand. | (see §3 below) |
| `vocab.py` | `Vocab` dataclass + `extract_vocab` + `validate_vocab`, both walking reconstructed trajectories. | §8 |
| `canonicalize.py` | `canonicalize_episode` — typed-local-id renumbering, deterministic or RNG-driven. | §11.7, METHOD §4.1.4 |
| `priors.py` | `BasePrior` ABC + `ZeroPrior` only. | §10 |
| `dataset.py` | `SpectreDataset`, `SpectreTrainingExample`, `SpectreBatch`, `collate_spectre_batch`. Online F-subset sampling and tensorization. | §11 |
| `eda.py` | All exploratory analysis — direct functions over loaded episodes (no parquet). | replaces §9 |
| `env_registry.py`, `envs/` | Local env registration (RoutedTransport2D in particular). | (no spec analogue) |

---

## 2. What matches the spec

- **Three-layer separation in spirit.** Raw episodes are the only persisted
  ground truth; training examples are sampled online by the Dataset and never
  materialized.
- **One file per episode**, gzip-pickle, atomic temp-then-rename writes
  (`io.atomic_write_pickle_gz`), with `.tmp` scrubbing on startup
  (`io.scrub_partial_writes`). Spec §3.2, §5.8, §6.5.
- **Collection config hashing.** `CollectionConfig.config_hash` is
  `sha256(canonical-JSON of all fields except created_at)[:12]` (§4.3) and is
  embedded into every `ProvenanceBlock`.
- **Non-short-circuiting refinement** with a stable per-skeleton seed
  (`_refinement_seed = blake2b-8(rule:problem_id:skeleton_idx)`) — §6.2, §6.3.
- **Three outcome categories.** `"success" / "fail" / "error"` are tracked
  separately; errors are excluded from both R and F at sample time
  (`SpectreDataset.__getitem__`). §5.6, §11.5.
- **Episode invariants I1–I4** asserted in `EpisodeRecord.__post_init__`.
- **Dataset invariants I8–I11** asserted in `SpectreDataset.__getitem__`. The
  exact set killed by Attempt 2 (F containing successes) is checked.
- **Filtering of non-trainable episodes** at Dataset init: `num_skeletons < 2`
  or `num_success == 0`. Filtered ids retained on `filtered_problem_ids`.
- **Vocab extraction from train only**, with `<OOV>` reserved at index 0,
  lexicographic ordering for determinism. `validate_vocab` returns a
  deduplicated findings list rather than raising — caller decides hard-fail.
- **Substage A storage**: only `s_0` and per-skeleton `final_abstract_state`
  are persisted. `state_path_depth = "s0_sL_only"` is the only supported value
  (`config.__post_init__` `NotImplementedError` on `"full"`).
- **Augmentation off for val/test** is a Dataset constructor argument
  (`augment: bool`), seeded from `(seed, index)`.
- **Worker-safe collection wrapper** (`collect_and_save_result`) for use under
  `multiprocessing` `spawn` — §6.6.

---

## 3. Key divergences from the spec

### 3.1 Layer 2 is collapsed: no parquet tables, no on-disk derived directory

- **No `manifest.parquet`.** Spec §7. There is no manifest-builder script.
  Episode paths are discovered by globbing `episodes/ep_*.pkl.gz`
  (`io.list_episodes`); summary fields are read by loading the episode itself.
  At our 500/100/100-per-env scale this is fast enough that the index is
  unnecessary; downstream code (`SpectreDataset.__init__`, EDA loaders) reads
  episodes once at startup.
- **No `train_flat.parquet` / `val_flat.parquet` / `test_flat.parquet`.**
  Spec §9. EDA in `eda.py` operates directly on the loaded `EpisodeRecord`
  list (`LoadedSplit`). Group-1 sanity, baselines (B1–B5), Δ/H bootstraps,
  and pass-bar evaluation are all functions over that in-memory structure.
  The flat-table abstraction was redundant once we accepted that EDA is
  a notebook driving Python functions, not SQL over parquet.
- **No `priors/<name>.parquet` cache.** Spec §10. `priors.py` ships only
  `ZeroPrior`, computed inline in `Dataset.__getitem__`. The `BasePrior`
  ABC keeps the swap-in interface stable for future HSR / PIGINet, but
  there is no parquet cache layer — when those land they will need a
  caching strategy (memory dict or parquet) added at that time.
- **No `derived/<env_variant>/` directory at all.** Vocab JSON lives wherever
  the caller writes it (typically alongside the env dir); there is no fixed
  `train_vocab.json` location enforced by code.
- **No `data/configs/collection_<hash>.yaml` written by the collector.**
  `collect.save_config_yaml` is provided but is opt-in; the entrypoint
  scripts under `experiments/` are the ones that call it.
- **No `runs/<run_id>/run_manifest.json`.** Spec §13. Hydra's run directory
  + git sha already capture this for our purposes; we did not add a separate
  pinning file.

### 3.2 Schema: live frozen-dataclass instances, not plain dicts

Spec §5.1 prescribes "plain Python types (str, int, float, bool, list, dict)
rather than live objects from `relational_structs`" so library refactors do
not break stored data. **We did not adopt this discipline.** `SkeletonRecord`
holds `tuple[GroundOperator, ...]`; `EpisodeRecord` holds
`RelationalAbstractState`, `frozenset[GroundAtom]`, etc. These pickle stably
because every constituent (`GroundOperator`, `GroundAtom`, `Object`, `Type`,
`Predicate`, `RelationalAbstractState`) is itself a frozen dataclass.

The justification (recorded in `schema.py`'s module docstring) is that the
plain-dicts requirement was insurance against pickle instability that has
not materialized in practice; carrying live objects lets downstream code
(canonicalization, vocab extraction, EDA baselines, `_remap_operator`) call
back into the substrate's APIs without a re-hydration step.

We also deliberately omit:
- `x0` (`ObjectCentricState`) — Φ never sees continuous state.
- `RelationalAbstractGoal` — its `state_abstractor` callable is not reliably
  pickleable. We store `goal_atoms: frozenset[GroundAtom]` instead.

### 3.3 Provenance gains a `scene_latent` slot

`ProvenanceBlock.scene_latent: Optional[dict[str, str]]` (default `None`)
holds env-specific per-episode context — currently the
`(blocked_color, blocked_grasp)` pair from RoutedTransport2D. This field
is not in spec §5.3. It was added so EDA can read the per-episode latent
without re-running the environment.

### 3.4 Config field naming is substrate-aligned

The spec §4.2 names `refinement_max_samples`, `refinement_wall_clock_cutoff_s`.
We use the names that match the actual `bilevel_planning` API:
- `num_sampling_attempts_per_step` (knob on `BacktrackingRefiner`)
- `refinement_timeout_s` (arg to `Refiner.__call__`)
- `abstract_plan_timeout_s` (new field; not in the spec but required for the
  pool generator)
- `heuristic_name` (default `"hff"`) and `max_trajectory_steps` likewise added.

`problem_seed_range` (a tuple) is split into two ints
`problem_seed_start` / `problem_seed_end` for cleaner YAML round-tripping.

### 3.5 Outcome record carries `refiner_metadata` instead of generic instrumentation

Spec §5.6 reserved `stuck_step_index` and `sampler_retries` (populated only
when `collect_instrumentation=true`). We keep both fields but the
`collect_instrumentation` flag is gated `NotImplementedError` in v0.1
(`config.__post_init__`). In its place, `OutcomeRecord.refiner_metadata:
dict[str, object]` is populated for RoutedTransport2D from
`ThreeGateRefiner.last_outcome` (`stuck_cause`, `stuck_op_name`,
`stuck_step_index`, modeled `wall_clock_s`). This was the simplest way to
get RT2D's structured failure causes into EDA without subclassing the
backtracking refiner.

### 3.6 Vocab extraction uses STRIPS progression, not just stored states

`SPECTRE_TRAINING_PIPELINE_SPEC.md` §8.3 implies scanning the stored atoms
(s_0 + final_abstract_state). We discovered that this misses predicates that
only ever live in intermediate states — `Holding(?robot, ?block)` in
ClutteredStorage2D being the canonical example, since Pick adds it and the
matching Place deletes it.

`trajectory.reconstruct_trajectory(s_0, operator_seq)` performs forward STRIPS
progression (`(state.atoms - op.delete_effects) | op.add_effects`) and is
called by both `vocab.extract_vocab` and `vocab.validate_vocab` to scan every
reachable state. As a side benefit, this also means future Substage B
upgrades require no schema change — intermediate states are recoverable
on demand.

### 3.7 Dataset includes tensorization (`collate_spectre_batch`, `SpectreBatch`)

Spec §11.8 says the collate function is a separate concern. We co-locate it
in `dataset.py`. The `SpectreBatch` dataclass pins exact tensor shapes to
the model spec (§4.1.3 of the method spec): `(B, R, L)`, `(B, R, L, A)`, etc.
Operator-arg local ids use `_local_id(p.name) + 1` so 0 reserves the pad slot
— a small contract that the collate and the encoder both honor.

`canonicalize.py` and `_local_id` together implement the typed-local-id
canonicalization the spec puts in §11.8: object names become `"{type}_{idx}"`
in `canonicalize_episode`, and the collate parses the suffix back to an int.

### 3.8 Episode-level LRU cache, not multi-worker-shared

`SpectreDataset` builds a per-instance `lru_cache(maxsize=64)` over
`load_episode`. Spec §11.4 left caching as "optional"; we always build it.
Each DataLoader worker gets its own cache; we do not share across workers.

### 3.9 Smoke-test protocol is implicit

Spec §12 lists a 7-step pre-collection smoke test. In practice we run the
EDA notebook on a pilot collection (a small `num_problems`) and gate full
collection on `evaluate_pass_bar` (`eda.py`) returning `primary_pass=True`.
The first six steps of §12 are unit-tested under `tests/approaches/spectre/`
rather than scripted as a single smoke runner.

---

## 4. Things that are not implemented at all

- All Layer-2 parquet outputs (manifest, flat tables, prior caches) — see §3.1.
- `validate_vocab.py` as a script — there is a `validate_vocab(...)` function
  in `vocab.py` but no CLI; callers wire it up in their entrypoint.
- `collection_log.jsonl` per-episode log line — Hydra's stdout is the log.
- `compute_priors.py` — there is no separate prior-population script because
  there is no prior other than zero.
- Run manifest for training reproducibility (spec §13). Hydra run dirs cover
  this.
- Any DAgger / cost-weighted-loss / OOV-fallback machinery (spec §15 deferred
  list) — also not implemented, as the spec itself defers these.

---

## 5. Migration-back-to-spec checklist

If the parquet layer becomes worth the cost (e.g. dataset grows past
~5k episodes per env, or collaborators want to query without booting the
substrate):

1. Add `build_manifest.py` walking `episodes/*.pkl.gz` and writing the schema
   from §7.2. Have `SpectreDataset.__init__` consume it instead of globbing
   + loading every file.
2. Add `build_flat_table.py` reading the vocab + episodes per §9.2. Re-point
   the EDA notebook at the parquet for stratified queries; keep the in-memory
   path for development.
3. Add `priors/<name>.parquet` keyed by `(problem_id, skeleton_idx, split)`
   (§10.2). Have `Dataset` read from the parquet instead of calling
   `prior.score(...)` per `__getitem__`.
4. Adopt the spec's plain-types schema only if a substrate refactor breaks
   pickle compatibility — until then, the live-object schema is cheaper to
   maintain.

---

*Last synced: 2026-04-24. If you change `schema.py`, `dataset.py`, or
`collect.py` in a way that affects the on-disk contract, update this
document alongside the code.*
