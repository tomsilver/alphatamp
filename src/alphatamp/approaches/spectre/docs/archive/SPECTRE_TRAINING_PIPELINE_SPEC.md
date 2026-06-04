# SPECTRE: Training Pipeline Specification

**Data collection, storage, and training-example construction**

*Version 0.1 — pre-implementation specification. Companion to `SPECTRE_METHOD_SPEC.md`. Sections marked* ⚠ *contain decisions that are provisional pending owner review or milestone validation.*

---

## 1. Purpose and Scope

### 1.1 What this document covers

This document specifies the end-to-end pipeline that produces, stores, and serves training data for SPECTRE. Concretely, it covers:

- The collection procedure that, for each planning problem, generates a candidate skeleton pool and records the refinement outcome of every skeleton.
- The on-disk layout of collected data, its serialization format, and the schema of derived artifacts used for exploratory analysis.
- The vocabulary extraction step that produces the fixed operator/predicate/type tables referenced by the skeleton encoder.
- The PyTorch `Dataset`/`DataLoader` interface that performs online F-subset sampling at training time.
- The invariants that must hold across these components, and the metadata required for reproducibility.

### 1.2 What this document does not cover

- The model architecture (Φ, Ψ, σ). See `SPECTRE_METHOD_SPEC.md` §4.
- The training loop, optimizer schedule, and loss implementation. See `SPECTRE_METHOD_SPEC.md` §5.3–5.5.
- The test-time inference loop and evaluation metrics. See `SPECTRE_METHOD_SPEC.md` §6 and the evaluation plan (separate document).
- DAgger-style on-policy correction. Deferred per `SPECTRE_METHOD_SPEC.md` §8.3.

### 1.3 Relationship to `SPECTRE_METHOD_SPEC.md`

This document refines `SPECTRE_METHOD_SPEC.md` §5.1 ("Data collection") into implementation-level detail. Where the two documents appear to disagree, the method specification is authoritative for *what* is collected and this document is authoritative for *how* it is collected and stored. Deferred decisions flagged in the method spec (e.g. Substage A vs. B, refiner instrumentation) are resolved here with explicit defaults and forward-compatibility hooks.

---

## 2. Architectural Principle: Three-Layer Separation

The pipeline is organized around a strict separation between three classes of artifacts. Keeping them on distinct layers — of disk, of code, and of regeneration cadence — is the single most important design decision in this document.

### 2.1 Layer 1: Raw episode records

One record per `(environment, split, problem_id)` triple. Contains the candidate pool, every skeleton's refinement outcome, and all provenance needed to reproduce collection. These records are ground truth: collected once, re-collected only on collection-config change.

### 2.2 Layer 2: Derived analysis artifacts

Flat parquet tables (one row per skeleton or per episode), vocabulary files, and prior-score caches. Cheap to regenerate from Layer 1; ephemeral in principle but materialized on disk for performance. Rebuilt whenever the derivation code changes.

### 2.3 Layer 3: Training examples

The `(R, SUCC ∩ R, F)` triples defined in `SPECTRE_METHOD_SPEC.md` §5.2. **Never materialized to disk.** Sampled online by the PyTorch `Dataset.__getitem__` from Layer 1 records. Pre-materializing training examples is explicitly rejected because:

- An episode with `|FAIL_e|` failures admits `2^|FAIL_e|` possible F subsets; enumeration is infeasible and random pre-materialization throws away the episode structure needed for different sampling strategies.
- The F-sampling distribution is a hyperparameter (uniform-over-subsets is the default; size-weighted, DAgger-corrected, and curriculum variants are anticipated).
- The choice of prior π is a hyperparameter and must be swappable without regenerating examples.

---

## 3. Directory Layout

All pipeline artifacts live under a single `data/` root. Environment variants (e.g. `ClutteredRetrieval2D-o10` vs. `-o25`) are treated as **distinct environments** with separate directories, because their object counts and induced skeleton distributions differ.

```
data/
  configs/
    collection_<hash>.yaml          # frozen collection config; hash in filename
  raw/
    <env_variant>/
      train/
        episodes/
          ep_00000.pkl.gz
          ep_00001.pkl.gz
          ...
        manifest.parquet             # one row per episode (see §7)
        collection_log.jsonl         # one line per collection attempt
      val/
      test/
  derived/
    <env_variant>/
      train_vocab.json               # operator/predicate/type vocab (see §8)
      train_flat.parquet             # EDA table (see §9)
      val_flat.parquet
      test_flat.parquet
      priors/
        hsr.parquet                  # (problem_id, skeleton_idx, prior_score)
        piginet.parquet
        zero.parquet                 # trivially all zeros; materialize for uniformity
  runs/
    <run_id>/
      run_manifest.json              # references all input artifacts (see §13)
      checkpoints/
      logs/
```

### 3.1 Naming conventions

- `<env_variant>` is a lowercase, hyphen-free identifier derived from the gym id: e.g. `clutteredretrieval2d_o10`, `clutteredstorage2d_b7`, `motion2d_p2`.
- Episode filenames are zero-padded 5-digit integers: `ep_{problem_id:05d}.pkl.gz`. Five digits accommodates up to 100,000 problems per split, well above the 500/100/100 budget.
- Config hash is the first 12 hex characters of a SHA-256 over the canonical-JSON-serialized collection config.
- Run ids are ISO-8601 timestamps with a short random suffix: `2026-04-22T14-30-00_a3f1`.

### 3.2 One episode per file

Collection writes one file per episode, not one aggregate file per split. This is deliberate:

- Resumption after interruption requires no index rewrite; existence of the file is the resumption check.
- Parallel collection workers never contend for the same file.
- Corrupting a single episode (e.g. a disk error mid-write) does not corrupt the split.

The cost is filesystem inode pressure, which is negligible at the 500–700 files per split scale.

---

## 4. Collection Configuration

### 4.1 Purpose

Before any collection code runs, the set of decisions that define "which dataset are we producing" is frozen into a single config object, serialized to YAML, hashed, and stored under `data/configs/`. The hash travels with every artifact downstream. Any change — env kwargs, seed range, refinement budget — produces a new hash and triggers fresh collection rather than polluting an old dataset.

### 4.2 Required fields

| Field | Type | Notes |
|---|---|---|
| `env_id` | str | Full gym id, e.g. `"kinder/ClutteredRetrieval2D-o10-v0"` |
| `env_kwargs` | dict | Passed to `create_bilevel_planning_models`, e.g. `{"num_obstructions": 10}` |
| `env_variant` | str | Derived directory name (see §3.1); must be deterministic from `env_id` |
| `package_versions` | dict | `{"bilevel_planning": ..., "kinder": ..., "kinder_models": ..., "relational_structs": ...}` resolved via `importlib.metadata` at config creation time |
| `split` | str | One of `"train"`, `"val"`, `"test"` |
| `num_problems` | int | 500 / 100 / 100 per `SPECTRE_METHOD_SPEC.md` §5.1 |
| `problem_seed_range` | tuple[int, int] | Inclusive-exclusive range. Convention: train=[0,500), val=[500,600), test=[600,700) |
| `K_max` | int | Hard cap on skeleton pool size per episode; default 50 per `SPECTRE_METHOD_SPEC.md` §7.2 |
| `refinement_max_samples` | int | Per-skeleton sampler retry budget |
| `refinement_wall_clock_cutoff_s` | float | Per-skeleton wall-clock cutoff |
| `refinement_seed_rule` | str | Identifier for the seed-derivation function; default `"v1_hash_problem_skeleton"` (see §6.3) |
| `collect_instrumentation` | bool | Whether to record refiner instrumentation (stuck-step, retries); default `false` per `SPECTRE_METHOD_SPEC.md` §8.2 |
| `state_path_depth` | str | `"s0_sL_only"` (default; see §5.3) or `"full"` (records every intermediate state) |
| `git_sha` | str | Commit of the collection code at config creation time |
| `created_at` | str | ISO-8601 timestamp |

### 4.3 Hashing rule

Hash is computed over the canonical JSON serialization with keys sorted, over *all* fields except `created_at`. This ensures bit-identical configs produce identical hashes regardless of creation time.

### 4.4 Split independence

Train, val, and test are separate config files (same `env_id`, different `split` and `problem_seed_range`). They share the same `env_kwargs`, `K_max`, and refinement budgets by convention but the code does not enforce this — an implementor must verify consistency across splits and flag any mismatch.

---

## 5. Episode Record Schema

### 5.1 Overview

An episode record captures everything about one `(problem, skeleton_pool, outcomes)` tuple. It is the unit of collection, the unit of storage, and the unit consumed by the PyTorch `Dataset`. The schema prioritizes stability over compactness: fields store plain Python types (str, int, float, bool, list, dict) rather than live objects from the `relational_structs` library, so that library refactors do not break stored data.

### 5.2 Top-level structure

An episode record is a dictionary with five blocks:

```
{
  "provenance":   { ... },
  "problem":      { ... },
  "skeleton_pool": [ skeleton_record, ... ],
  "outcomes":     [ outcome_record, ... ],   # parallel to skeleton_pool by index
  "summary":      { ... }
}
```

`skeleton_pool` and `outcomes` are parallel lists indexed by `skeleton_idx`. They are kept separate (rather than fused per skeleton) so that outcome re-collection — e.g. averaging over refiner seeds for evaluation — can overwrite one without touching the other.

### 5.3 Provenance block

```
provenance: {
  "problem_id":       int,          # matches filename's ep_{N:05d}
  "env_id":           str,
  "env_variant":      str,
  "split":            str,
  "config_hash":      str,          # 12-char hex
  "problem_seed":     int,          # seed used to generate this problem
  "git_sha":          str,
  "collection_timestamp": str,      # ISO-8601
  "package_versions": dict
}
```

### 5.4 Problem block

```
problem: {
  "initial_state":      <serialized ObjectCentricState>,
  "initial_abstract_state": {
    "atoms":   list of [pred_name: str, arg_names: list[str]],
    "objects": list of [obj_name: str, type_name: str]
  },
  "goal_atoms": list of [pred_name: str, arg_names: list[str]],
  "object_registry": {
    obj_name: type_name for every object in the problem
  }
}
```

The `object_registry` is a convenience: every `arg_name` appearing anywhere in the record resolves here to a type. This lets downstream code reconstruct live `Object` instances without consulting the environment.

`initial_state` serialization: use whatever mechanism `relational_structs.ObjectCentricState` provides for round-trip; if none exists natively, implement a `to_dict`/`from_dict` in a collection-side adapter module. Do not pickle the live object.

### 5.5 Skeleton record

One entry per skeleton in the pool. Per the Substage A decision (see §5.3 of `SPECTRE_METHOD_SPEC.md` resolution below), intermediate abstract states are **not** stored. Only the operator sequence, the initial abstract state (shared with the episode-level `problem.initial_abstract_state`, not replicated), and the final abstract state `s_L` are kept.

```
skeleton_record: {
  "skeleton_idx":    int,           # 0 .. K-1, canonical order from symbolic planner
  "operator_seq":    list of {
                       "op_name":  str,              # lifted operator name
                       "arg_names": list[str]        # ground object names; resolve via object_registry
                     },
  "final_abstract_state": {
    "atoms":   list of [pred_name, arg_names],
    "objects": list of [obj_name, type_name]
  }
}
```

**Rationale for omitting intermediate states.** The intermediate `s_1, ..., s_{L-1}` are deterministic functions of `(s_0, g_1, ..., g_i)` under STRIPS semantics and therefore recoverable at encode time if ever needed. Storing them would increase episode size without adding information. `s_0` is retained because it encodes problem-instance information beyond what the operator sequence conveys (object counts, initial configuration). `s_L` is retained because, while it entails the goal, its full atom set is not determined by the goal alone and may carry weak signal for future ablations or goal-conditioning experiments.

**Forward compatibility with Substage B.** If future work requires intermediate states, the collection config's `state_path_depth` field switches to `"full"` and the skeleton record gains an `"intermediate_abstract_states"` list. Models that were trained on Substage A data remain valid; models trained on Substage B data require `state_path_depth == "full"` in the underlying records and should assert this at Dataset init.

### 5.6 Outcome record

One entry per skeleton, parallel to `skeleton_pool` by index.

```
outcome_record: {
  "skeleton_idx":            int,          # must match skeleton_record's
  "outcome":                 str,          # "success" | "fail" | "error"
  "refinement_wall_clock_s": float,        # inclusive of failed attempts within budget
  "refinement_seed":         int,          # deterministic, see §6.3
  "stuck_step_index":        int | None,   # populated iff collect_instrumentation=true
  "sampler_retries":         int | None,   # populated iff collect_instrumentation=true
  "error_info":              dict | None,  # populated iff outcome=="error"; contains exception class and message
  "refiner_metadata":        dict          # free-form; reserved for future fields
}
```

**Three outcome categories, not two.** `"success"` and `"fail"` are the semantic outcomes of refinement. `"error"` is reserved for unexpected exceptions (environment crashes, sampler bugs, timeout with undefined state). Error outcomes must never be treated as failures for training: they are filtered out by the `Dataset` and surfaced in the EDA table for debugging. Conflating errors with failures is the single easiest way to introduce silent training-data corruption.

### 5.7 Summary block

Redundant with `outcomes` but useful for manifest construction and fast filtering.

```
summary: {
  "num_skeletons":        int,
  "num_success":          int,
  "num_fail":             int,
  "num_error":            int,
  "first_success_idx":    int | None,   # smallest idx with outcome=="success", or None
  "total_wall_clock_s":   float,
  "pool_truncated":       bool          # true iff symbolic planner produced > K_max skeletons
}
```

### 5.8 Serialization

Episode records are serialized as gzip-compressed pickle. Because all fields are plain Python types (no live `GroundOperator` or `RelationalAbstractState` instances), pickle compatibility is not tied to library versions.

- File extension: `.pkl.gz`.
- Compression level: 6 (Python default); files are small and compression throughput is not the bottleneck.
- Write atomically: to `ep_NNNNN.pkl.gz.tmp`, `fsync`, rename.

Alternative formats (msgpack, JSON) were considered and rejected for version 0.1: pickle is sufficient given the plain-types discipline, and switching format later is a trivial reader change. Do not block on this decision.

---

## 6. Collection Procedure

### 6.1 Top-level algorithm

For each `problem_id` in the config's seed range:

1. Skip if `episodes/ep_{problem_id:05d}.pkl.gz` exists and its embedded `config_hash` matches the current config hash.
2. Instantiate the environment with `problem_seed = problem_id` (or a deterministic function thereof — fix and document the convention).
3. Build the `SesameModels` via `create_bilevel_planning_models(observation_space, action_space, **env_kwargs)`.
4. Extract initial state, compute `s_0`, derive goal atoms.
5. Invoke the symbolic planner to obtain the full candidate pool `S`. If `|S| > K_max`, truncate to the first `K_max` in the planner's canonical order and set `summary.pool_truncated = true`.
6. For each `s` in `S`, in the planner's canonical order, call the refiner with a deterministic per-skeleton seed and record the outcome. **Attempt every skeleton regardless of whether an earlier one succeeded** (see §6.2).
7. Assemble the episode record. Write atomically.
8. Append one line to `collection_log.jsonl` with the summary block plus timing.

### 6.2 Non-short-circuiting refinement

`SPECTRE_METHOD_SPEC.md` §5.1 specifies that refinement is attempted on every skeleton in the pool, regardless of whether some earlier skeleton has already succeeded. This is explicitly different from the standard TAMP planner behavior, which stops at the first success.

**Implementation note.** Depending on the `bilevel_planning` library's API, this may require either:

- Calling a lower-level refiner directly (per-skeleton), bypassing the planner's loop; or
- Invoking the planner in a mode that disables early termination.

Budget meaningful time for this integration. It is the single most likely place for a subtle bug to slip in. A collection-time assertion — `assert len(outcomes) == len(skeleton_pool)` — catches the most common failure mode.

### 6.3 Refinement seeding

Each per-skeleton refinement attempt must use a deterministic seed derived from `(problem_id, skeleton_idx)`. The default rule is:

```
refinement_seed = hash_u64(f"{config.refinement_seed_rule}:{problem_id}:{skeleton_idx}")
```

where `hash_u64` is a stable 64-bit hash (e.g. `int.from_bytes(hashlib.blake2b(..., digest_size=8).digest(), "big")`). The stability is essential for reproducibility: re-running collection with the same config must produce identical outcomes for deterministic refiners.

### 6.4 Error handling

Any exception raised by the refiner is caught, its class and message recorded in `outcome.error_info`, and `outcome.outcome` set to `"error"`. Collection continues with the next skeleton. An episode with any `"error"` outcomes is not inherently invalid — it is flagged in the manifest (`num_error > 0`) and can be re-collected or filtered depending on downstream needs.

If the *symbolic planner* fails (no skeletons producible for the problem), the episode record is still written, with an empty `skeleton_pool` and `outcomes`, and `summary.num_skeletons == 0`. Downstream code treats these as non-trainable but diagnostically useful.

### 6.5 Resumption and atomic writes

- Resumption: before collecting problem `i`, check for the existence of `ep_{i:05d}.pkl.gz`. If present, read just the `provenance.config_hash` field (pickle supports partial loads via the usual unpickling shortcut; if this proves fragile, maintain a small sidecar `ep_{i:05d}.hash` file with just the hash). If hash matches, skip; if not, delete and re-collect with a warning.
- Atomic writes: always write to `.tmp`, `fsync`, rename.
- Partial-write detection: on startup, scan for any `.tmp` files and delete them; they represent a previous process killed mid-write.

### 6.6 Parallelization

Episodes are independent. The default driver is a single-process loop for simplicity. A `multiprocessing.Pool`-based parallel driver is a straightforward extension:

- Use `spawn` start method (not `fork`) to avoid environment-creation issues with C extensions.
- One worker per physical core; the 2D environments are CPU-bound.
- Workers do not share file handles for episode files; contention is zero.
- The `collection_log.jsonl` is appended to by multiple workers; use `fcntl.flock` or a single-writer logger process to avoid interleaved lines.

### 6.7 Observability

- `collection_log.jsonl` appended at end of each episode; format: one JSON object per line with fields `{problem_id, num_skeletons, num_success, num_fail, num_error, total_wall_clock_s, timestamp}`.
- A separate progress log written every N episodes (N=10) with running averages: helpful for `tail -f` during long collections.
- `tqdm` for interactive progress. When running under nohup or in a batch system, redirect tqdm to the progress log.

---

## 7. Manifest

### 7.1 Purpose

`manifest.parquet` provides a fast, scannable index over the episodes in a split. It is the answer to questions like "which episodes succeeded?", "what's the distribution of pool sizes?", "how much wall-clock total?" — without opening any episode file.

### 7.2 Schema

One row per episode. Columns are the provenance block and summary block, flattened:

| Column | Type | Source |
|---|---|---|
| `problem_id` | int | provenance |
| `env_variant` | str | provenance |
| `split` | str | provenance |
| `config_hash` | str | provenance |
| `problem_seed` | int | provenance |
| `collection_timestamp` | str | provenance |
| `num_skeletons` | int | summary |
| `num_success` | int | summary |
| `num_fail` | int | summary |
| `num_error` | int | summary |
| `first_success_idx` | int (nullable) | summary |
| `total_wall_clock_s` | float | summary |
| `pool_truncated` | bool | summary |
| `file_path` | str | computed (relative to raw split dir) |

### 7.3 Generation

The manifest is built by a separate `build_manifest.py` script that scans `episodes/*.pkl.gz` and emits the parquet. It is always regenerable; it is not part of the atomic-write path of individual episode collection. Rebuild after collection completes, and after any re-collection.

---

## 8. Vocabulary Extraction

### 8.1 Purpose

The skeleton encoder Φ (`SPECTRE_METHOD_SPEC.md` §4.1) requires fixed-size embedding tables keyed by lifted operator names, predicate names, and object types. These tables are extracted from the training split only, frozen, and used unchanged at validation/test time.

### 8.2 Vocabulary file schema

`derived/<env_variant>/train_vocab.json`:

```
{
  "config_hash":           str,   # train split's config hash
  "operators":             { op_name: idx for op_name ∈ observed },
  "predicates":            { pred_name: { "arity": int, "idx": int } },
  "types":                 { type_name: idx },
  "max_operator_arity":    int,   # observed max across all ground operators
  "max_predicate_arity":   int,
  "max_skeleton_length":   int,
  "max_atoms_per_state":   int,
  "max_objects_per_state": int,
  "max_pool_size":         int,
  "max_objects_per_type":  { type_name: int }
}
```

The `max_*` fields are observed maxima, useful as sanity checks and as default tensor-size hints. They are not used to enforce bounds — the model is designed to handle variable sizes — but they surface distributional surprises (e.g. a test-time pool 10× larger than any seen in training).

### 8.3 Extraction procedure

Scan every episode in `raw/<env_variant>/train/episodes/`. For each:

- Add every `op_name` to `operators`.
- Add every `pred_name` (with its arity) to `predicates`.
- Add every `type_name` to `types`.
- Update maxima.

Indices are assigned in insertion order (sorted lexicographically for determinism: sort the set of observed names before enumerating). `operators["<OOV>"] = 0` and equivalent for predicates and types are reserved slots even if the graceful-fallback feature (`SPECTRE_METHOD_SPEC.md` §8.5) is not yet implemented; this way the slot exists when that feature is added.

### 8.4 Validation against val/test

A separate `validate_vocab.py` script scans `val/` and `test/` episodes and asserts every observed operator/predicate/type is in `train_vocab.json`. Any OOV entry triggers a loud failure with the offending names listed. Per `SPECTRE_METHOD_SPEC.md` §7.2, OOV is unsupported in version 0.1; this script is what enforces the assumption.

---

## 9. Flat EDA Table

### 9.1 Purpose

One parquet file per split with one row per `(problem_id, skeleton_idx)`. All columns are scalars or short strings. This is what gets loaded into pandas for exploratory data analysis: success rate stratified by skeleton length, wall-clock distribution by operator count, correlation between pool size and first-success index, and so on.

### 9.2 Schema

| Column | Type | Description |
|---|---|---|
| `problem_id` | int | |
| `skeleton_idx` | int | |
| `num_skeletons_in_pool` | int | K for this episode |
| `outcome` | str | `"success"` / `"fail"` / `"error"` |
| `refinement_wall_clock_s` | float | |
| `skeleton_length` | int | Number of operators in the sequence |
| `skeleton_is_first_success` | bool | True iff this idx == episode's first_success_idx |
| `operator_name_sequence` | str | `"|"`-joined op names; useful for groupby |
| `num_unique_operators` | int | |
| `num_objects_in_s0` | int | |
| `num_atoms_in_s0` | int | |
| `num_objects_in_sL` | int | |
| `num_atoms_in_sL` | int | |
| `stuck_step_index` | int (nullable) | |
| `sampler_retries` | int (nullable) | |
| `count_<op_name>` | int | One column per lifted operator in vocab; count of occurrences in the skeleton |

The `count_<op_name>` columns are generated dynamically from `train_vocab.json` and are present in all three splits' flat tables (val and test may have zero counts for some columns, which is fine).

### 9.3 Generation

A `build_flat_table.py` script reads the vocab file and all episodes, emits the parquet. Runs in seconds for the full budget. Always regenerable from Layer 1.

---

## 10. Prior Score Caching

### 10.1 Purpose

Static priors π(s) (HSR, PIGINet, zero) are deterministic functions of `(problem, skeleton)` and are reused across every training epoch and across every F-subset sampled from a given episode. They are expensive enough (especially PIGINet, a neural net) to justify caching on disk.

### 10.2 Schema

`derived/<env_variant>/priors/<prior_name>.parquet`:

| Column | Type |
|---|---|
| `problem_id` | int |
| `skeleton_idx` | int |
| `split` | str |
| `prior_score` | float |

Keyed by `(problem_id, skeleton_idx, split)` — joining a single prior table across all splits is cheaper than managing one file per split, given the small row counts (~5k × 3 per env).

### 10.3 Population

A separate `compute_priors.py` script is invoked once per prior variant per environment. It reads the raw episodes, invokes the prior's scoring function, writes the parquet. Re-run whenever the prior implementation changes; the `derived/` directory is disposable.

For `zero`, the table is trivially populated. It is materialized (not special-cased in the Dataset) so that the Dataset code path is uniform across prior choices.

---

## 11. PyTorch Dataset: Online F-Subset Sampling

### 11.1 Design

The `Dataset` consumes Layer 1 episode records and produces Layer 3 training examples. It is where F-subset sampling happens.

- `__len__` returns the number of *trainable* episodes in the split (see §11.5 for filtering).
- `__getitem__(i)` loads episode `i`, samples a subset `F ⊆ FAIL_e`, constructs `R = S \ F`, and returns the training example.

### 11.2 Construction arguments

```
SpectreDataset(
    raw_dir:          Path,       # points to raw/<env_variant>/<split>/
    manifest_path:    Path,       # derived from raw_dir but explicit for clarity
    vocab_path:       Path,       # path to train_vocab.json (same for all splits)
    prior_table_path: Path,       # path to derived/<env_variant>/priors/<name>.parquet
    f_sampling:       str,        # "uniform_subsets" (default); other modes reserved
    augment:          bool,       # whether to apply within-type object renumbering (see §11.7)
    split:            str,        # for prior lookup and assertion
    seed:             int         # RNG seed for F sampling and augmentation
)
```

### 11.3 `__getitem__` contract

Given episode `e`:

1. Load `e` from disk (with optional LRU cache; see §11.4).
2. Compute `SUCC_e = [idx for idx, o in enumerate(e.outcomes) if o.outcome == "success"]` and `FAIL_e = [... "fail"]`. Skeletons with outcome `"error"` are excluded from both sets.
3. Sample `F ⊆ FAIL_e`. For `f_sampling="uniform_subsets"`: include each `idx ∈ FAIL_e` independently with probability 0.5. This is uniform over the power set of `FAIL_e`, including `F = ∅` and `F = FAIL_e`.
4. Compute `R = {0..K-1} \ F \ {error_indices}`. Equivalently, `R` is the union of `SUCC_e` and `FAIL_e \ F`.
5. Look up prior scores for each `idx ∈ R` from the prior table.
6. Assemble the training example.

### 11.4 Training example structure

```
training_example: {
  "problem_id":      int,
  "initial_abstract_state": { ... },   # s_0, shared across all skeletons in R and F
  "goal_atoms":      [ ... ],
  "object_registry": { ... },
  "R_skeletons":     [ skeleton_record, ... ],   # |R| entries
  "R_priors":        list[float],                # |R| entries, parallel to R_skeletons
  "R_success_mask":  list[bool],                 # True iff skeleton succeeded in original episode
  "F_skeletons":     [ skeleton_record, ... ],   # |F| entries; failures only
}
```

The Dataset returns *structured data*, not tensors. A separate `collate_fn` (see §11.8) converts to padded tensors. This separation is deliberate: the Dataset stays pure-Python and easy to inspect; tensor construction is a pipeline stage that can be swapped.

### 11.5 Filtering of non-trainable episodes

At Dataset init, filter out episodes where:

- `summary.num_skeletons == 0` (symbolic planner failure).
- `summary.num_skeletons == 1` (nothing to rank).
- `summary.num_success == 0` (PL loss undefined: `Z_plus = 0` for all F choices).

Report the count of filtered episodes at init. Filtered-out problem ids are retained in a `self._filtered_problem_ids` attribute for reporting.

Episodes containing some `"error"` outcomes but still satisfying the above are retained; the error skeletons are simply excluded from both `SUCC_e` and `FAIL_e`.

### 11.6 Invariants (asserted in `__getitem__`)

These are cheap runtime assertions that catch the exact bug class that killed Attempt 2 in `SPECTRE_METHOD_SPEC.md` §5.2.

```
assert set(F_indices).issubset(fail_indices_set), \
       "F must contain only failed skeletons (per SPECTRE spec §5.2)"
assert set(R_indices).issuperset(succ_indices_set), \
       "R must contain all successful skeletons"
assert set(R_indices).isdisjoint(F_indices), \
       "R and F must be disjoint"
assert len(R_indices) + len(F_indices) + len(error_indices) == num_skeletons
```

### 11.7 Data augmentation

Per `SPECTRE_METHOD_SPEC.md` §4.1.4, training applies random within-type object renumbering as augmentation. The Dataset applies this at `__getitem__` time when `augment=true`:

- Gather all object names referenced in the episode (operator args + atom args + object_registry keys).
- Group by type (from `object_registry`).
- Within each type, draw a random permutation of the within-type indices.
- Rewrite every `arg_name` consistently across the returned structure.

Augmentation is off for val and test Datasets. Augmentation RNG is seeded from `(seed, problem_id, epoch)` so that re-running produces reproducible augmentations within an epoch.

### 11.8 Collate function

A separate `collate_fn(batch)` converts a list of training examples into padded tensors. Key responsibilities:

- Pad `R_skeletons` and `F_skeletons` across the batch to the max `|R|` and max `|F|` respectively; emit attention masks.
- Pad each skeleton's operator sequence to the batch-max skeleton length.
- Replicate `initial_abstract_state` per-skeleton as needed for the encoder (this is the "`s_0` replicated as a token inside every skeleton" detail from the method spec — the data layout stores it once, the model input layout replicates it).
- Convert operator names, predicate names, and type names to integer indices via the vocab.
- Apply typed-local-id canonicalization (`SPECTRE_METHOD_SPEC.md` §4.1.4): convert concrete object names to `(type_idx, within_type_idx)` pairs using a canonical ordering.

Canonicalization is applied inside `collate_fn` (not in `__getitem__`) so that the Dataset output remains human-inspectable.

### 11.9 Expected access pattern

- Train DataLoader: `shuffle=True`, `num_workers=4`, `persistent_workers=True`. Multiple epochs reuse the same Dataset; each epoch re-samples F subsets fresh.
- Val/test DataLoader: `shuffle=False`, `augment=False`. ⚠ For validation AUROC(t) computation (`SPECTRE_METHOD_SPEC.md` §9 terminology), a separate non-sampling iterator is needed that enumerates specific F sizes deterministically. Specify this when implementing the validation metric.

---

## 12. Smoke-Test Protocol

Before launching full collection, execute the following sequence on a single environment:

1. **Pilot collection.** Collect 5 episodes with the chosen config. Verify files exist, decompress, have the expected schema.
2. **Manual inspection.** Pretty-print 2 episodes. Cross-check: does `summary.num_success + summary.num_fail + summary.num_error == summary.num_skeletons`? Does `first_success_idx` point at a skeleton with `outcome == "success"`? Does `operator_seq` reference only objects in `object_registry`?
3. **Manifest.** Build the manifest. Confirm row count = 5 and schema is correct.
4. **Vocab (on the pilot).** Extract a vocab from just the 5 pilot episodes. Confirm the set of operator names matches what is expected from the environment's `create_bilevel_planning_models` (by manual inspection of the env file).
5. **Flat table.** Build the flat table. Confirm row count = sum of pool sizes, column schema is correct, no null values in non-nullable columns.
6. **Dataset smoke.** Instantiate the Dataset (with a trivial zero prior), draw 10 random training examples, verify for each that the four invariants in §11.6 hold. Print one example in full for manual sanity.
7. **Scale check.** Benchmark the average wall-clock per episode over the 5 pilots. Multiply by 500 × 3 splits × 5 envs. If the projection is more than ~72 hours of single-machine wall-clock, revisit before committing.

Only after all seven steps pass cleanly does full collection launch.

---

## 13. Run Manifest (Training-Side Reproducibility)

### 13.1 Purpose

Every training run writes a `run_manifest.json` that pins every input artifact to a specific version. Six months from now, the question "which dataset produced figure 3?" must have an unambiguous answer.

### 13.2 Schema

```
{
  "run_id":            str,
  "start_time":        str,   # ISO-8601
  "git_sha":           str,   # training code
  "dataset": {
    "env_variants":    list[str],
    "config_hashes": {
      "<env_variant>": { "train": str, "val": str, "test": str }
    },
    "vocab_paths":     { "<env_variant>": str },
    "prior_name":      str,
    "prior_table_paths": { "<env_variant>": str }
  },
  "model_config":      dict,           # full model hyperparameters
  "optimizer_config":  dict,
  "seeds": {
    "model_init":      int,
    "data_sampling":   int,
    "f_subset":        int
  },
  "hardware": {
    "gpu":             str,
    "cuda_version":    str
  }
}
```

Written at the start of training, not the end. On a crashed or interrupted run, the manifest alone identifies what was being attempted.

---

## 14. Invariants (Consolidated Reference)

These must hold at all times. Violations indicate data corruption and should trigger loud failures.

**Collection-time:**

- I1. `len(episode.outcomes) == len(episode.skeleton_pool)`.
- I2. For every `i`, `episode.outcomes[i].skeleton_idx == episode.skeleton_pool[i].skeleton_idx == i`.
- I3. `summary.num_success + summary.num_fail + summary.num_error == summary.num_skeletons`.
- I4. If `summary.first_success_idx is not None`, then `episode.outcomes[first_success_idx].outcome == "success"`.
- I5. `provenance.config_hash` matches the config hash recorded in `data/configs/`.

**Vocabulary:**

- I6. Every operator/predicate/type appearing in val/test is present in `train_vocab.json`.
- I7. `train_vocab.json.config_hash` matches the train split's config hash.

**Dataset (per `__getitem__`):**

- I8–I11. The four invariants in §11.6.
- I12. Every `skeleton_idx` in `R` or `F` has a corresponding entry in the prior table.

**Priors:**

- I13. Every `(problem_id, skeleton_idx)` pair in any raw episode has exactly one row in each prior table for that env variant.

---

## 15. Deferred / Explicitly Out of Scope for v0.1

- **DAgger on-policy correction.** Per `SPECTRE_METHOD_SPEC.md` §8.3, implement only if M8 evaluation reveals an offline-vs-rollout gap.
- **Refiner instrumentation.** The config flag `collect_instrumentation` is plumbed, but the default is `false`. Enabling it requires refiner-side changes outside the scope of this document.
- **Cost-weighted loss variant.** `SPECTRE_METHOD_SPEC.md` §8.4. The `refinement_wall_clock_s` field is collected in all cases, so switching to the weighted variant is a loss-function change only — no data pipeline change needed.
- **Graceful OOV fallback for vocabulary.** `SPECTRE_METHOD_SPEC.md` §8.5. Reserved `<OOV>` slot exists in the vocab; assertion-based hard failure remains the default.
- **Cross-environment joint datasets.** Each environment variant is a separate directory. Joint training across environments is a Dataset-layer composition task, not a pipeline-layer concern.
- **Compressed columnar skeleton storage.** Parquet-of-structs or Arrow tables for skeleton pools. Pickle is sufficient at the 500-episode scale; revisit only if storage or IO become a bottleneck.
- **Skeleton-level deduplication across episodes.** Structurally identical skeletons from different problems are not currently deduplicated. Revisit only if the vocab or model shows evidence of overfitting to high-frequency skeleton templates.

---

## 16. Open Questions and Owner Decisions

### 16.1 Confirmed (resolved for v0.1)

- **Substage A as the data contract.** Per owner guidance, most of the signal is expected to come from `s_0`; intermediate states are deterministic and need not be stored. `s_L` is retained for future flexibility. `state_path_depth = "s0_sL_only"` is the default config value.
- **Refiner instrumentation default off.** `collect_instrumentation = false`.
- **Error outcomes are a third category**, not conflated with failure.

### 16.2 Requires owner confirmation before full collection

- ⚠ **Problem-seed convention.** Train=[0,500), val=[500,600), test=[600,700). Verify this matches any pre-existing split conventions in `kinder` environments to avoid accidental overlap with prior experiments.
- ⚠ **Refinement wall-clock cutoff.** A numeric default must be chosen and justified. The cutoff directly caps dataset collection time and implicitly defines the failure distribution.
- ⚠ **`K_max` setting per environment.** 50 is the `SPECTRE_METHOD_SPEC.md` §7.2 default, but environments with naturally smaller pools may warrant a lower cap to avoid wasted attempts; environments with larger natural pools may warrant a higher cap. Measure pool-size distributions in the pilot collection (§12) before freezing.

### 16.3 Revisit at M4/M8

- Whether to upgrade serialization to msgpack or columnar parquet.
- Whether to materialize a `train_flat_with_embeddings.parquet` including fixed 64-d skeleton embeddings from a trained Φ, for fast iteration on Ψ-only experiments.
- Whether the `augment=true` path should apply multiple augmentations per epoch per episode (oversampling).

---

*End of training pipeline specification.*
