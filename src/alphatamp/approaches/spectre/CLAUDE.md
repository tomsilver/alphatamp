# SPECTRE — Project Context

Authoritative context for the SPECTRE project (branch `adaptive` lineage).
Monorepo-general conventions live in the root `CLAUDE.md`; everything
spectre-specific lives here and in the imported docs:

@docs/proposal.md
@docs/decisions.md

## What this is

**Direction pivot, 2026-06-25 (see [`docs/proposal.md`](docs/proposal.md) §0 and
[`docs/decisions.md`](docs/decisions.md) 2026-06-25).** SPECTRE's contribution is
now a **representation question** for plan-feasibility prediction in
fully-observable, deterministic bilevel TAMP: *what should a feasibility
predictor represent skeletons and problems over?* The hypothesis (falsifiable, not
proven) is that a richer-than-pixels, cheaper-than-full-state representation
predicts refinement feasibility more sample-efficiently and with weaker perception
than a low-level (PIGINet-style) predictor over the concrete initial state — with
a **crossover** in the low-data / weak-perception regime (efficiency, not
information access). *Abstract-first* is the current leading candidate, one point
in a design space (learned latents, object-centric/graph features, invented
predicates, …). The **adaptive skeleton re-ranker** described below is now a
**secondary, composable** increment, not the headline: our own ablation
attributes only ~27% of the margin over B4 to failure-conditioning, the static
representation the rest (`docs/notebook.md` 2026-06-06/2026-06-25).

Mechanically, the SPECTRE re-ranker is a learned model for bilevel TAMP: given a
pool of candidate skeletons and the set of skeletons that have already failed
refinement, it picks the next skeleton to try. Candidate method = SPECTRE;
baselines are B1–B5 (random, default order, static-historical,
adaptive-historical, oracle) — never describe spectre-specific code or labels
as a "baseline". Re-ranker metric: mean time-to-first-success vs B4, mean ± std
over ≥ 3 seeds, evaluated uncensored at attempt budget 30 (= the candidate-pool
cap, so the budget never binds; `decisions.md` 2026-06-07). Model selection
(`val_rollout_attempts`) stays at its own budget 20. RoutedTransport2D-n3-v1
(in-package) is the bespoke env behind the re-ranker results; under the pivot,
evaluation prefers **pre-existing environments meeting the representation-
advantage property wishlist** (`docs/proposal.md` §0), with bespoke still in
scope. ClutteredStorage2D-b5/b7 and StickButton2D-b5 collections are historical.

## Where everything lives

| Piece | Path |
|---|---|
| Package (model, dataset, collection, EDA) | `src/alphatamp/approaches/spectre/` — do not move; it IS `alphatamp.approaches.spectre` |
| RT2D environment | `src/alphatamp/approaches/spectre/envs/routedtransport2d/`, registered via `env_registry.py` |
| DD2D environment (migrated) + JSON→EpisodeRecord converter | `src/alphatamp/approaches/spectre/envs/dd2d/` (env + raw_v2 dataset + `MIGRATION_DD2D.md`); `spectre_operators.py` (drawer substrate) + `spectre_convert.py` (converter). Wired as env_variant `dd2d_v2`, **not** a native SesameModels env — see `docs/decisions.md` 2026-07-12 |
| Docs (living proposal, ADR log, lab notebook, lit review, archived specs + dated writeup snapshots) | `src/alphatamp/approaches/spectre/docs/` |
| Hydra entry points + configs + SLURM launchers + analysis notebook | `experiments/spectre/` (configs under `experiments/spectre/conf/`) |
| Tests | `tests/approaches/spectre/` (RT2D env tests under `envs/routedtransport2d/`) |
| Data (gitignored) | `data/spectre/{raw,derived,checkpoints,configs}/` — the `data_root: "data/spectre"` convention is relative to the repo root |
| SLURM logs | `experiments/slurm_outputs/` (shared scratch, gitignored) |

Spectre's Hydra configs are self-contained: `experiments/spectre/conf/`
holds `spectre_collect.yaml`, `spectre_build_vocab.yaml`, `spectre_train.yaml`,
`dd2d_convert.yaml`, the spectre-only env group
(`env/{clutteredstorage2d_b5,routedtransport2d_n3_v1,stickbutton2d_b5,dd2d_v2}.yaml`),
and spectre's own `hydra/launcher/slurm.yaml` (8 cpus / 32 GB). The shared
`experiments/conf/` tree belongs to other projects — never put spectre configs
there.

## Compute resources (dev workstation)

Primary dev/training box as of 2026-07-18 (replaces the earlier MacBook M3 Pro /
MPS setup; the SLURM launchers below remain for cluster runs):

- **GPU — NVIDIA RTX 5090, 32 GB VRAM, Blackwell (sm_120).** Driver 595.71,
  CUDA 13.2 runtime; driver-only, no `nvcc`/CUDA toolkit (fine for prebuilt
  wheels). Single GPU → one training run at a time; multi-seed sweeps run
  sequentially or share the card. Training goes on CUDA — unlike the old
  CPU/MPS box, so watch for code that hard-codes `cpu`/`mps` or mixes devices.
- **PyTorch must be the cu130 build** (`torch==2.13.0+cu130`), installed with
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`
  **before** `uv pip install -e ".[develop,ttd]"`. cu130 is the
  actively-maintained line with native sm_120 support and matches the CUDA 13.2
  driver. `pyproject.toml` keeps `torch` unpinned (shared with SLURM / other
  machines), so the cu130 index is applied at install time, not baked in — if an
  editable reinstall ever pulls a PyPI-default torch, re-run the cu130 install
  and re-verify with a real device op (`(x@x)` on `cuda`), not just
  `torch.cuda.is_available()`.
- **CPU — AMD Ryzen 9 9950X, 16 cores / 32 threads** (~5.75 GHz boost). Local
  data collection / worker-parallel stages (`spectre_collect.py`, EDA) can use
  far more workers than the SLURM launcher's 8 cpus / 32 GB.
- **RAM ~64 GB** (59 GiB usable) + 14 GiB swap. **Disk ~1.2 TB free** on `/` for
  datasets/checkpoints under `data/spectre/`.
- **OS/toolchain:** Ubuntu 26.04 LTS, Python 3.11 venv (`.venv`, uv-managed),
  uv 0.11.29. Substrate dep pins were modernized on 2026-07-18
  (`decisions.md` that date) to resolve on a fresh machine — kindergarden 0.2.0,
  prpl-mono `e215d1fc`, kinder-baselines `4c731dc8`.

## Pipeline & how to run

Always `source .venv/bin/activate` first; run from the repo root. Stages in
order (details in @docs/proposal.md §4–5; respect the de-risking gates):

1. **Collect** (500 train / 100 val / 100 test per env):
   `python experiments/spectre/spectre_collect.py env=routedtransport2d_n3_v1 split=train problem_seed_start=0 problem_seed_end=500`
   — or `bash experiments/spectre/collect_routedtransport2d_n3_v1.sh` (all
   three splits locally), or `sbatch experiments/spectre/spectre_collect.slurm …`
   / `bash experiments/spectre/submit_spectre_<env>.sh` on a cluster.
   **DD2D (`env=dd2d_v2`) skips this stage:** it has no native SPECTRE env — run
   `python experiments/spectre/dd2d_convert.py` instead to convert the migrated
   `envs/dd2d/data/dd2d/raw_v2` JSON dataset into `data/spectre/raw/dd2d_v2/…`
   episodes. To generate *fresh* DD2D data, run DD2D's own collector
   (`python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.collect --out-root …`,
   needs shapely + the planners) and re-run the converter pointed at its output.
   Stages 2–4 below then work unchanged with `env=dd2d_v2`.
2. **Vocab** (train split only, OOV-checks val/test):
   `python experiments/spectre/spectre_build_vocab.py env=routedtransport2d_n3_v1`
3. **Sanity-check** the collection + one collated batch:
   `python experiments/spectre/spectre_check_pipeline.py env=routedtransport2d_n3_v1`
4. **Train** (multi-seed):
   `python experiments/spectre/spectre_train.py env=routedtransport2d_n3_v1 seed=0`
   — or `sbatch --array=0-2 experiments/spectre/spectre_train.slurm` (one seed
   per array task; extra Hydra overrides forwarded).
5. **Analyze / experiments:** `experiments/spectre/analyze_spectre.py`
   (a marimo notebook — run with `marimo edit experiments/spectre/analyze_spectre.py`;
   drives `eda.py`: EDA gates, B1–B5 brackets, rollout simulation, comparison
   table). Diagnostics:
   `python experiments/spectre/spectre_probe_atom_sensitivity.py env=routedtransport2d_n3_v1 seed=0`.

## Conventions and invariants

- **Loss:** listwise Plackett-Luce only. Pointwise BCE killed Attempt 2.
- **F-subsets:** `F ⊆ FAIL_e` strictly — never successes in F.
- **Vocab from train only;** id 0 = `<PAD>`/`<OOV>`; local-id 0 = pad.
- **Augmentation:** training only; per-type policy from `env_registry.py`
  (ordered/semantic RT2D types are non-augmentable).
- **Metrics:** model selection and early stopping are rollout-based —
  `val_rollout_attempts` (simulated sparse rollout on val, attempt budget 20;
  `checkpoint_metric` in `train.py`) — chosen to align with the rollout-based
  test-time objective. AUROC(3) is a secondary offline diagnostic (drives the
  during-training de-risking gates), never the selection criterion. The
  D.1/D.2 atom-sensitivity probes do NOT predict rollout performance —
  diagnostics only, never optimization targets.
- **Reporting:** every number is mean ± std over ≥ 3 seeds.
- **Doc updates are part of development** — see "Documentation discipline"
  below. Archived specs and snapshots in `docs/archive/` are frozen — never
  edit them; annotations go in `docs/archive/README.md`.
- Tests: `pytest tests/approaches/spectre/` (slow tests skipped by default;
  `-m ""` to include).

## Documentation discipline — keep the living docs alive

The living docs are the project's research memory: code records *what*, the
docs record *why*. Commit messages and checkpoint dirs are not a research
log. After any change that is more than mechanical, decide — **before
committing** — which of these needs an entry, and ship the entry **in the
same commit** as the change:

| The work... | Update | Format |
|---|---|---|
| produced any run/EDA/probe/ablation numbers — **including failed and negative runs** | `docs/notebook.md` | dated What / Result / Takeaway-next entry (format at top of file) |
| chose between alternatives with lasting consequences, killed an approach, or changed a convention / invariant / metric / protocol | `docs/decisions.md` | ADR: context → decision → consequences, newest first |
| changed the method, loss, architecture, data pipeline, or evaluation protocol | `docs/proposal.md` | edit in place — it must always describe the *current* method; also reconcile §6 (add new unknowns, remove resolved ones) |

Exempt: mechanical refactors, formatting, typo fixes, CI appeasement —
anything that cannot affect results or future decisions.

Litmus test: *"In 3 months, will we know this happened and why?"* If the
change could alter a number in a future writeup/snapshot, or a future
contributor could plausibly re-litigate the choice, it needs an entry.

This rule exists because the passive version of it failed: `notebook.md`
stayed empty for the project's first ~2 months of training runs, and the
load-bearing rollout-metric decision (`b74b593`) got its ADR only
retroactively. Write entries at change time, not archaeology time.
