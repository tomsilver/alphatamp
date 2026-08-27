# SPECTRE — Code Map (for paper-writing)

A stage-by-stage index of the source files that make up the SPECTRE pipeline, from data
collection through model/training to evaluation. Intended as an orientation index when reading the
code for the write-up; the authoritative method description is [`../as_built.md`](../as_built.md).

Two path prefixes keep this compact:

- **`pkg/`** = `src/alphatamp/approaches/spectre/`
- **`exp/`** = `experiments/spectre/`

---

## The core pipeline (data → model → train → eval)

### Shared data structures & primitives (used by every stage)

- `pkg/schema.py` — `EpisodeRecord`, `ObjectGeometry`, pool/candidate records
- `pkg/failure_record.py` — `FailureRecord` (culprits, dev_blame, state_delta)
- `pkg/canonicalize.py` — object → typed-local-id canonicalization (⚠ not idempotent; always tensorize from raw)
- `pkg/tags.py` — episode-local object tags
- `pkg/trajectory.py` — STRIPS progression over a stored skeleton
- `pkg/io.py` — atomic gzip-pickle episode IO
- `pkg/config.py` — `CollectionConfig` (frozen, hashable, YAML round-trippable)
- `pkg/vocab.py` — operator/predicate/type vocab (train-only)

### Stage 1 — Data collection

- `pkg/collect.py` — collection driver (`refiner_mode` = analytic / real / hybrid-prune)
- Entry points:
  - `exp/spectre_collect.py` — native-env collection
  - `exp/dd2d_convert.py` — DD2D JSON → `EpisodeRecord`
  - `exp/sb2d_collect.py` — StickButton2D (pooled variants, reject-and-resample)
  - `exp/restock3d_v3_collect.py` + `exp/restock3d_v3_run_all.sh` — Restock3D-v3 (per-stratum jobs)
  - `exp/spectre_build_vocab.py` — vocab build

### Stage 2 — The model (the core contribution)

- `pkg/model.py` — `SpectreModel`, `SpectreConfig`, **`ResidualEvidenceScorer`** (the X2 residual)
- `pkg/encoders.py` — Scene / Candidate / Record / AtomProfile / PointSet encoders
- `pkg/layers.py` — SAB / PMA / attention primitives
- `pkg/domain.py` — `DomainSpec` / `QueryAxioms` (`step_certificate`, the per-environment contract)
- `pkg/unified_evidence.py` — **coverage / waste / repeat** definitions
- `pkg/facts.py` — typed-fact / hint context
- `pkg/dataset.py` — training-example construction / tensorization (`build_example`)
- `pkg/loss.py` — Plackett-Luce listwise loss
- `pkg/priors.py` — plug-in static priors

### Stage 3 — Training

- `pkg/train.py` — training loop (two-stage residual-adaptive: freeze + warm-start)
- Entry points:
  - `exp/spectre_sweep.py` — concurrent multi-arm / multi-seed sweeps (presets)
  - **`exp/refresh_dd2d_sb2d_train.sh`** — the deployed DD2D + SB2D two-stage recipe
  - `exp/sb2d_finalize.sh` — SB2D vocab → bracket → train → score
  - `exp/restock3d_v3_train.sh` — Restock3D-v3 (jointly-trained `+repeat`)

### Stage 4 — Inference / evaluation / comparison

- `pkg/inference.py` — deployed rollout + `load_checkpoint`
- `pkg/compare.py` — metric engine (`rollout_fp`, loaders, bootstrap)
- `pkg/compare_envs.py` — the `EnvSpec` registry (one entry per environment)
- `pkg/eda.py` — B1–B5 baseline brackets + EDA gates
- `pkg/dp_on_counts.py` — the B6 DP-on-counts baseline
- Entry points:
  - `exp/spectre_score.py` — paired-bootstrap scoring vs a baseline
  - `exp/precompute_dd2d_cache.py` — stamp the per-problem FP cache the notebook reads
  - `exp/compare_methods.py` — the marimo method-comparison notebook
  - `exp/spectre_status.py` — long-run status/ETA · `exp/spectre_run.sh` — logged run wrapper

---

## Baselines (comparison methods)

- **PIGINet** (low-level predictor): `pkg/baselines/piginet/{model,train,dataset,encoders,losses,eval,tokenize,record,domain}.py` + `{dd2d,sb2d,restock}_adapter.py`
- **LAZY** (learned adaptive, Khodeir et al.): `pkg/baselines/lazy/{model,train,dataset,graph,feasibility,rollout,tree,eval,domain}.py`
- **VLMPlan** (zero-shot VLM): `pkg/baselines/vlmplan/{loop,adapter,models,parsing,registry,score,template,runio}.py` + `{dd2d,sb2d,restock}_adapter.py` + `sb2d_label.py`; entry points `exp/vlmplan_{run,score}.py`

---

## Environments (the three evaluated)

- **DD2D** (Drawer-Declutter 2D, packing/retrieval)
  - adapters: `pkg/envs/dd2d/spectre_{convert,operators,geometry,harvest,render}.py`
  - env internals: `pkg/envs/dd2d/drawer/{grasps,refine,problem,scene,shapes,planning,enumerate,label,certificate,world,record_ext}.py`
- **StickButton2D** (tool-use button-pressing): `pkg/envs/stickbutton2d/{heuristic,scene_geometry,sampler,instrumented_refiner,geometry,strata,diagnostics,render}.py`
- **Restock3D-v3** (3D kinematic-PyBullet shelf packing): `pkg/envs/restock3d/{feasibility_v3,generator_v3,strata_v3,models_v3,place_controller_v3,oracle_v3,plan_generator_v2,models_v2,place_controller_v2,kinematic_env,instrumented_refiner,scene_geometry,section_geometry,region_geometry,generator,render}.py` (`generator.py` and the `_v2` files are imported by v3)

---

## Docs — the writing source (most valuable for the paper)

- `pkg/docs/proposal.md` (framing / §0), **`as_built.md`** (the method write-up), `restock3d_proposal.md`, `research_lit.md`, `porting_guide.md`
- `pkg/docs/paper-writing/Problem_Methodology.md`
- **`pkg/docs/decisions/`** (README + `01`–`07`) and **`pkg/docs/notebook/`** (README + `01`–`07`) — the ADR log + results record

  *(The intermediate/dev docs — fix & feature proposals, autonomous session logs, and the built-feature guides that are now folded into `as_built.md` — were moved to a local, gitignored `pkg/docs/archive/` and are not part of the synced repo.)*

---

## Skip for paper-writing (noise / non-source)

- **`data/`** — datasets, checkpoints, videos (large; not source)
- **Retired / reference envs**: `pkg/envs/shelf3d/`, `pkg/envs/restock3d/front-grasp-tall-block*/`
- **Scratch experiment scripts**: `exp/restock3d_{probe,harness,kmax,blocking_sweep,clutter_sweep,calibrate,gates,...}.py`, all `exp/restock3d_v2_*` and v1 stage scripts, `exp/shelf3d_*`, `exp/{ablation_*,holdout_vs_full,fc2_build_phi,w2_sweep,failed_records_sufficiency}.py`
- `pkg/envs/dd2d/drawer/{demo*,eda*,heuristic*,inspect_example,render_families}.py` — dev/scratch
- `pkg/docs/archive/` — a **local, gitignored** folder (superseded specs + archived intermediate dev docs: fix/feature proposals, autonomous session logs, built-feature guides); not in the synced repo. Also skip `tests/` (optional).

---

## Where to start reading

For the write-up, the method and the numbers are concentrated in: `pkg/docs/` (esp. `as_built.md`,
`decisions/07`, `notebook/07`) + `pkg/model.py` + `pkg/dataset.py` + `pkg/unified_evidence.py` +
`pkg/compare.py` / `pkg/compare_envs.py`.
