# SPECTRE — Project Context

Authoritative context for the SPECTRE project (branch `adaptive` lineage).
Monorepo-general conventions live in the root `CLAUDE.md`; everything
spectre-specific lives here and in the imported docs:

@docs/proposal.md
@docs/decisions.md

## What this is

SPECTRE is a learned **adaptive skeleton re-ranker** for bilevel TAMP: given a
pool of candidate skeletons and the set of skeletons that have already failed
refinement, it picks the next skeleton to try. Candidate method = SPECTRE;
baselines are B1–B5 (random, default order, static-historical,
adaptive-historical, oracle) — never describe spectre-specific code or labels
as a "baseline". Headline metric: mean time-to-first-success vs B4, mean ± std
over ≥ 3 seeds, attempt budget 20. Primary evaluation env:
RoutedTransport2D-n3-v1 (in-package); ClutteredStorage2D-b5/b7 and
StickButton2D-b5 collections are historical.

## Where everything lives

| Piece | Path |
|---|---|
| Package (model, dataset, collection, EDA) | `src/alphatamp/approaches/spectre/` — do not move; it IS `alphatamp.approaches.spectre` |
| RT2D environment | `src/alphatamp/approaches/spectre/envs/routedtransport2d/`, registered via `env_registry.py` |
| Docs (living proposal, ADR log, lab notebook, lit review, archived specs) | `src/alphatamp/approaches/spectre/docs/` |
| Hydra entry points + configs + SLURM launchers + analysis notebook | `experiments/spectre/` (configs under `experiments/spectre/conf/`) |
| Tests | `tests/approaches/spectre/` (RT2D env tests under `envs/routedtransport2d/`) |
| Data (gitignored) | `data/spectre/{raw,derived,checkpoints,configs}/` — the `data_root: "data/spectre"` convention is relative to the repo root |
| SLURM logs | `experiments/slurm_outputs/` (shared scratch, gitignored) |

Spectre's Hydra configs are self-contained: `experiments/spectre/conf/`
holds `spectre_collect.yaml`, `spectre_build_vocab.yaml`, `spectre_train.yaml`,
the spectre-only env group (`env/{clutteredstorage2d_b5,routedtransport2d_n3_v1,stickbutton2d_b5}.yaml`),
and spectre's own `hydra/launcher/slurm.yaml` (8 cpus / 32 GB). The shared
`experiments/conf/` tree belongs to other projects — never put spectre configs
there.

## Pipeline & how to run

Always `source .venv/bin/activate` first; run from the repo root. Stages in
order (details in @docs/proposal.md §4–5; respect the de-risking gates):

1. **Collect** (500 train / 100 val / 100 test per env):
   `python experiments/spectre/spectre_collect.py env=routedtransport2d_n3_v1 split=train problem_seed_start=0 problem_seed_end=500`
   — or `bash experiments/spectre/collect_routedtransport2d_n3_v1.sh` (all
   three splits locally), or `sbatch experiments/spectre/spectre_collect.slurm …`
   / `bash experiments/spectre/submit_spectre_<env>.sh` on a cluster.
2. **Vocab** (train split only, OOV-checks val/test):
   `python experiments/spectre/spectre_build_vocab.py env=routedtransport2d_n3_v1`
3. **Sanity-check** the collection + one collated batch:
   `python experiments/spectre/spectre_check_pipeline.py env=routedtransport2d_n3_v1`
4. **Train** (multi-seed):
   `python experiments/spectre/spectre_train.py env=routedtransport2d_n3_v1 seed=0`
   — or `sbatch --array=0-2 experiments/spectre/spectre_train.slurm` (one seed
   per array task; extra Hydra overrides forwarded).
5. **Analyze / experiments:** `experiments/spectre/analyze_spectre.ipynb`
   (drives `eda.py`: EDA gates, B1–B5 brackets, rollout simulation, comparison
   table). Diagnostics:
   `python experiments/spectre/spectre_probe_atom_sensitivity.py env=routedtransport2d_n3_v1 seed=0`.

## Conventions and invariants

- **Loss:** listwise Plackett-Luce only. Pointwise BCE killed Attempt 2.
- **F-subsets:** `F ⊆ FAIL_e` strictly — never successes in F.
- **Vocab from train only;** id 0 = `<PAD>`/`<OOV>`; local-id 0 = pad.
- **Augmentation:** training only; per-type policy from `env_registry.py`
  (ordered/semantic RT2D types are non-augmentable).
- **Metrics:** model selection is rollout-based on val; AUROC(3) is the
  offline metric that predicts test attempts. The D.1/D.2 atom-sensitivity
  probes do NOT predict rollout performance — diagnostics only, never
  optimization targets.
- **Reporting:** every number is mean ± std over ≥ 3 seeds.
- Record run outcomes in `docs/notebook.md`; lasting decisions in
  `docs/decisions.md`; method changes in `docs/proposal.md`. Archived specs in
  `docs/archive/` are historical — do not edit them.
- Tests: `pytest tests/approaches/spectre/` (slow tests skipped by default;
  `-m ""` to include).
