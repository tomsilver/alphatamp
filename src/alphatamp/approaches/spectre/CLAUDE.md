# SPECTRE — Project Context

Authoritative context for the SPECTRE project (branch `adaptive` lineage).
Monorepo-general conventions live in the root `CLAUDE.md`; everything
spectre-specific lives here and in the imported docs:

@docs/proposal.md
@docs/decisions/README.md

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
cap, so the budget never binds; [`decisions.md` 2026-06-07](docs/decisions/README.md)). Model selection
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
| DD2D environment (migrated) + JSON→EpisodeRecord converter | `src/alphatamp/approaches/spectre/envs/dd2d/` (env + raw_v2 dataset + `MIGRATION_DD2D.md`); `spectre_operators.py` (drawer substrate) + `spectre_convert.py` (converter). Wired as env_variants `dd2d_v2` and `dd2d_v3` (the re-collection after the 2026-07-24 grasp changes), **not** a native SesameModels env — see `docs/decisions.md` 2026-07-12 |
| **StickButton2D** — the second evaluation environment | `src/alphatamp/approaches/spectre/envs/stickbutton2d/` — thin adapters over kinder's own env and refiner: `heuristic.py` (geometry-aware A* + the acyclic pool filter), `scene_geometry.py`, `sampler.py`/`instrumented_refiner.py` (class-2 evidence), `strata.py` (pooled-variant problem ids), `geometry.py`, `diagnostics.py`. Collected as env_variant **`stickbutton2d_v1`** (b1/b2/b3/b5 pooled, button count = stratum; b10 dropped). Entry points `experiments/spectre/sb2d_{collect,baselines,rerank_gate}.py` + `sb2d_finalize.sh` |
| VLMPlan baseline (zero-shot VLM planner) | `src/alphatamp/approaches/spectre/vlmplan/` — env-agnostic core + `dd2d_adapter.py` as the only env-aware module; entry points `experiments/spectre/vlmplan_{run,score}.py`. Protocol: `docs/decisions.md` 2026-07-24; prompt deviations: `vlmplan/prompts/PROVENANCE.md` |
| Docs (living proposal, lit review, archived specs + dated writeup snapshots) | `src/alphatamp/approaches/spectre/docs/` |
| **ADR log** and **lab notebook** — chaptered by era, newest first | `docs/decisions/` and `docs/notebook/`, each with a **generated `README.md`** (chapter list, full entry ledger, by-track index, ID-resolution table, do-not-quote ledger, legacy date→entry map). Pre-split monoliths frozen in `docs/archive/*_monolithic.md`; `docs/decisions.md` / `docs/notebook.md` are stubs. Tooling: `doclog.py` + `experiments/spectre/decisions_index.py` |
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
6. **VLMPlan baseline** (zero-shot VLM comparison row; two stages, only the first needs a
   model, so a re-collection re-runs just the second):
   ```bash
   lms server start   # or any OpenAI-compatible server (vLLM)
   export OPENAI_BASE_URL=http://localhost:1234/v1 OPENAI_API_KEY=lm-studio
   python experiments/spectre/vlmplan_run.py   env=dd2d_v3 split=train n_problems=5 run=pilot
   python experiments/spectre/vlmplan_score.py env=dd2d_v3 split=train n_problems=5 run=pilot
   ```
   One `cache_subdir` is one method row — give a different `run` its own `cache_subdir`
   or the rows get averaged together (guarded, not silent). Check the printed
   **label-agreement gate** before trusting any number: below ~0.95 means the env code
   moved since that collection and in-pool vs off-pool labels disagree.

## SPECTRE v3 (in progress, 2026-07-26)

Migration from v2.2 to v3 per [`docs/SPECTRE_v3_proposal.md`](docs/SPECTRE_v3_proposal.md),
run as gated increments. **Current substrate is `dd2d_v4`** (grasp-fixed *and*
refiner-instrumented); dd2d_v2/v3 numbers predate the double-canonicalization fix and must
not be quoted without regenerating.

**Gate status.** Done: G0–G6b as before; **G7** (P-v3-3 falsified — `cand_overlap` is
load-bearing, −5.07 FP); **G8/P2–P9** the performance push (below). **G9 descoped** (encoder
built, experiment not run — its premise does not hold on DD2D: s0–s2 pools already contain
9-operator plans while s3 needs 7, so the position table is never OOV). **G10 not
attempted.** Remaining: **G11 consolidation** — `as_built_v3.md` and `porting_guide.md` are
written; `./run_ci_checks.sh` has residual pylint line-length debt.

**Where v3 stands (2026-07-31).** Deployed config =
`--overlap-mode jaccard --coverage-feats --aggregate-records --evidence-attn
--state-delta` (sweep preset `v3final`, checkpoints `checkpoints_v3_unified`), **with
proof-demotion off** and **coverage/waste on the unified definitions** — the latter is now
`TrainV3Config.unified_coverage=True`, so the preset needs no extra flag. Over **3 seeds
each**:

| | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| v3 deployed | **5.78 ± 0.10** | 0.00 | **3.44 ± 1.36** | **10.49 ± 0.77** | **9.19 ± 0.76** |
| v2.2 yardstick | 17.27 ± 3.02 | 0.00 | 13.67 ± 14.20 | 23.45 ± 2.76 | 31.95 ± 5.62 |

**≈ −11.5 FP vs v2.2** (the paired CI for that pair is not computed — `spectre_score_v3`
cannot take a v2 arm as `--baseline`). Against the *previous* v3 deployed definition the
paired margin is **−1.66 FP, CI [−2.71, −0.71]**, every seed beating every baseline seed
([`decisions.md`](docs/decisions/06-v3-performance.md) 2026-07-31). Reproduce:

```bash
python experiments/spectre/spectre_sweep.py --preset v3final --seeds 0 1 2
# demotion is OFF by default; `--with-demotion` is the ablation. The flag is GLOBAL, so
# `--v2-arm` would also lose its demotion -- the 17.27 yardstick comes from the compare
# cache, whose v2 path always demotes.
python experiments/spectre/spectre_score_v3.py --env-variant dd2d_v4 \
    --arm "v3 deployed:checkpoints_v3_unified" --seeds 0 1 2
```

**Pre-2026-07-31 v3 numbers are not retracted** — 7.44 and everything before it were
measured on a consistent definition and remain valid for what they measured. The checkpoint
decides which definition is used at scoring time (`inference_v3._emit_kwargs` reads
`unified_coverage` from the saved cfg; absent ⇒ the old formula), so old checkpoints keep
scoring correctly. `--legacy-coverage` retrains the old definition into a `_legacycov`
directory. The pre-memoization replicate of the unified arm is
`checkpoints_v3_unified_prememo` (5.83 ± 0.11).

**v3 is a purely learned ranker as of 2026-07-30** (`decisions.md`). Proof-tier demotion
was **cut from the method**: nothing outside the network touches the ordering, and
`apply_demotion=False` is the default in `deployed_rollout_v3_traced`, `cache_spectre3`,
`train_v3.deployed_val_fp` and `spectre_score_v3.py`. It cost **0.23 FP** (7.20 → 7.44);
it fired on only **6%** of deployed rollouts against **55%** on the stripped floor arm, and
the learned features absorbed ~79% of its value. The machinery is **kept and one flag away**
(`apply_demotion=True`) because the deduction is sound — on a domain whose proofs fire more
often the trade reverses. Two things bit during that change and will bite again:
**re-cache with `--force`** (`_dir_complete` keeps a stale dir), and **pin
`test_v3_equivalence` to `apply_demotion=True`** or it compares two different policies and
silently stops testing equivalence.

**Three numbers to quote together, not separately** (`as_built_v3.md` §7.1):
- **3 seeds is the count every *method* has** (v2.2 was trained at exactly 3), not the
  count v3 has — v3 has **6**, and over all six it reads **8.54 ± 1.43**.
- **The yardstick is v2.2's 3-seed mean (17.27), not its best seed (14.66).** With both
  sides at 3 seeds the like-for-like comparison is mean-to-mean; ~2.6 FP of the −9.83 is
  that change of basis, not a change in v3. v2.2's s1 spread is ±14.20 (seed 2 lands at
  30.04, `relrank` picking a bad epoch) — the miscalibration R8 replaced.
- **v2.2 keeps its demotion while v3 gives its up**, so the margin is measured against a
  stronger baseline than v3 gives itself. Deliberate: re-scoring the published baseline to
  match a v3 design choice would be moving the goalposts.

**s1 is where a small-seed report is least trustworthy.** The pre-delta arm's s1 read
3.79 ± 3.29 at 3 seeds and 5.60 ± 3.06 at 6, where four of six seeds were *worse* than the
v2.2 seed-0 figure. On a wide-spread stratum, check the margin against the seed sd, not the
sign.

**Both record consumptions matter** (6 seeds each): dropping the per-failure token stream
costs 1.28 FP (7.90 → 9.18), *entirely at s1* (5.60 → 10.78, worse than v2.2 there) while
s2/s3 tie, and it doubles the variance. Compact features carry s2/s3; tokens carry s1.

**What carries it: observed `coverage`/`waste`.** These are §5.1's necessity features with
per-object necessity **observed** (`FailureRecord.culprits`) instead of **predicted** — so no
head, no second loss, no geometry routine, and *more* C2-legal than the cut version since
nothing is inferred by us. Not `clears` (L2): that was a routine *we* ran. Both features are
exactly zero at |F|=0, so the first attempt stays static and the signal accrues as the
rollout observes. A leakage audit returned 0 violations.

**Retracted, do not quote:** G6's levels (18.59/19.15/20.95 — censored selector) and G6's
−3.37 "record increment", which was `cand_overlap`, not records (its bar removed both).

**Traps this push added** (details in `docs/autorun_decisions.md` A1–A13):
- **`dead` is a length proxy** — right at s3, wrong at s1 (corr(dead,|S|) = −0.284). Tuning
  it only trades strata; give the model the count it proxies for.
- **A token stream the model ignores is not free** — records cost −0.83 FP in training while
  `suppress_records` showed the deployed model barely reads them (16.17 → 16.40).
- **Evidence competed with geometry in one softmax** (~10 scene tokens vs up to 2045 record
  tokens), so discarding it was loss-minimizing.
- **Two runs sharing a checkpoint dir** silently interleave writes; `train_v3` now refuses
  via a `.owner` marker.

- **New modules** (v1/v2 are frozen — D-7): `domain.py` (the whole per-environment
  contract: per-query `QueryAxioms(monotone, local, exact)` + `min_calls_per_schema`),
  `failure_record.py`, `proof_demotion_v3.py`, `model_v3.py`, `dataset_v3.py`,
  `inference_v3.py`, `train_v3.py`, `necessity.py` (built, **unwired**).
- **D-8, exact-absence:** every v3 feature is config-gated and *off* reproduces v2.2's
  state dict byte-for-byte, so `test_v3_equivalence.py` keeps loading the v2.2 checkpoint
  and asserting identical decisions. That oracle is what makes data-path rewrites safe;
  it runs in `permissive` mode and retires when the position encoding changes (G9).
- **Necessity conditioning was CUT** ([`decisions.md` 2026-07-26](docs/decisions/README.md)): D2 showed the s2 deficit
  is *within-length*, which it does not address. P-v3-1 is withdrawn; s2 is reported as a
  characterized limitation.
- **Untaken lead:** enumeration order is a strong within-length signal (astar
  length-oracle 5.80 at s2) the deployed model cannot see, because R1 dropped the prior
  wholesale when only its short-first column was implicated. An index-only prior was never
  separately ablated.

Findings and numbers live in `docs/notebook.md` / `docs/decisions.md` under 2026-07-26 —
cite them rather than restating figures.

## StickButton2D — the second environment (2026-08-01)

DD2D was the only evaluation environment; SB2D is the second, so the generality claim in
[`docs/porting_guide.md`](docs/porting_guide.md) is finally being *tested* rather than
asserted. Everything wraps kinder's own env, operators and `BacktrackingRefiner`.
Chapter [`07-stickbutton2d`](docs/decisions/07-stickbutton2d.md) holds the ADRs.

**The dataset is `stickbutton2d_v1`**: b1/b2/b3/b5 pooled into one env_variant with
**button count as the stratum** (b10 dropped — 0/20 solvable, and the cause is pool prefix
homogeneity needing diverse plan *generation*, not a better heuristic). Problem ids encode
`split · 10⁶ + slot · 250000 + index` precisely so the existing `dd2d_compare.stratum_of`
returns the slot, which is what lets ~15 call sites work unchanged. **Strata are
contiguous pid bands, so *stride, never truncate* is load-bearing here** — `paths[:N]` is
all b1.

**The two strata that matter are b3 and b5.** Pool sizes after the acyclic filter are
≈2 / 6–34 / 200 / 200, so b1 and b2 are anchors that every method ties on (b1 static
reads 0.08 FP) — the shape DD2D's `s0 = 0.00` already has. Do not read a pooled "ALL"
mean over unbalanced strata as a method comparison.

**Headline (3 seeds, uncensored, test n=100).** Mean failed attempts before first success:

| | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| **SPECTRE v3** | **1.69 ± 0.26** | 0.08 | 0.24 | **1.13 ± 0.12** | **5.29 ± 1.04** |
| B3 static-historical (best baseline) | 6.41 | 0.08 | 0.36 | 9.84 | 15.36 |
| B2 default A* order | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |
| B4 adaptive-historical | 22.56 | 0.08 | 0.24 | 26.88 | 63.04 |
| B1 random | 21.04 | 0.24 | 5.22 | 47.79 | 30.90 |

**Three things to quote together, not separately** — all in
[`notebook/07`](docs/notebook/07-stickbutton2d.md):

- **B4 is worse than random here** (22.56 vs 21.04). SPECTRE's headline comparison on
  RT2D/DD2D is the adaptivity premium over B4; on SB2D the bar is **B3, a *static*
  ranker**, and any cross-environment framing has to say so.
- **b5's train split has only 17 episodes** (collection was cut at a wall-clock budget),
  so 5.29 at b5 is substantially a *generalisation* number, not a like-for-like stratum
  result. Re-measure before quoting it as one.
- **`waste` helps the model and hurts a rule.** As a hand-coded tie-break it degrades b3
  (4.44 vs 2.88 for coverage alone); as a learned column it is worth +0.36 FP, CI
  [+0.08, +0.67]. A failed non-learned re-ranking probe is therefore **not** grounds for
  dropping a feature — it tests monotone usability in the guessed direction, not
  information content.

**Three things about SB2D that are not true of DD2D**, each of which changed code:

- **It has no class-1 evidence at all.** kinder's collision check returns a bool, so every
  failure is a class-2 *deviation* between predicted and achieved abstract state. Both
  channels are now always wired and an empty one is provably inert; see the ADR.
- **`jaccard` and `dead` are constant across the b5 pool.** Every b5 plan has length 6
  (5 presses + 1 stick pickup), so `manipulated = {robot, stick}` for every candidate.
  Coverage/waste and the token structure are the *only* discriminating features there.
- **Its pool generator pads plans with `PickStick`/`PlaceStick` cycles.** Filtered out;
  near-inert at b5, decisive at b1 (200 candidates → 2).

**`spectre_collect.py` is not the collector here** — `experiments/spectre/sb2d_collect.py`
is, because the collection pools four kinder env ids into one variant and **rejects and
resamples** problems with no feasible skeleton. Post-collection, `sb2d_finalize.sh` runs
vocab → check → re-ranking gate → B1–B5 bracket → training → scoring in the order the
gates require.

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
- **Reporting:** paper numbers are mean ± std over ≥ 3 seeds. **Development runs
  1 seed** (2026-07-26 directive) — with 1 seed "within seed noise" is not
  measurable, so a gate is accepted by a **paired bootstrap over problems**
  (`spectre_score_v3.py`, the instrument the P1/P4/P5 gates used); pairing removes
  the between-problem variance that dominates here.
- **Doc updates are part of development** — see "Documentation discipline"
  below. Archived specs and snapshots in `docs/archive/` are frozen — never
  edit them; annotations go in `docs/archive/README.md`.
- Tests: `pytest tests/approaches/spectre/`. Slow tests are skipped by default and
  **`-m ""` does NOT include them** — `tests/conftest.py` overrides an empty
  markexpr back to `not slow`. Use `-m slow` to run them.

## Working practices (hardware, long runs, traps)

**Use the hardware. Parallelise whenever tasks are independent.** Training here is
**CPU-bound, not GPU-bound** — measured 79% tensorization / 21% GPU, and three
concurrent arms occupy 3.5 GB of the 5090's 33.7 GB. Run arms, seeds, ablations and
data collection *concurrently* rather than in sequence whenever they do not depend on
each other; serial runs leave both the GPU and ~30 CPU threads idle.

- `python experiments/spectre/spectre_sweep.py --preset g6` — concurrent arms, one log
  each. `--arm "name:args"` for ad-hoc arms, `--seeds 0 1 2` for the paper runs.
- Keep `max_parallel × (1 + num_workers)` under the core count (32) or the runs contend
  and wall-clock stops improving. Measured: 38.9 s/epoch serial → ~33 s/epoch with three
  arms at once (~3.4× throughput).
- The DD2D collector already parallelises via `--workers`.

**Long runs must be interruptible and must expose an ETA.** Anything over a few
minutes goes to a named log with periodic heartbeats, so progress and remaining time can
be checked at any moment without disturbing the run — and so a run that has clearly gone
wrong can be stopped early instead of discovered at the end.

- Launch via `spectre_run.sh <name> <cmd...>` (or `spectre_sweep.py`), which logs to
  `data/spectre/logs/<name>.log`.
- Check with `python experiments/spectre/spectre_status.py` (`--watch` to follow):
  what is running, latest heartbeat + ETA per job, recently finished checkpoints.
- When adding a long-running script, emit a periodic heartbeat with elapsed, progress
  and ETA, and state the expected total up front.

**GPU contention:** LM Studio / `llama-server` (the VLMPlan backend) holds ~30 GB of
VRAM and will starve training into CUDA OOM warnings. Stop it before a sweep; the
VLMPlan results are already cached under `compare_cache/vlmplan_*`, so nothing is lost.

**Traps that have each cost real time:**
- **Stride, never truncate.** Episodes are stored in seed order and the collector fills
  strata in seed bands, so `paths[:N]` yields only the easy strata. Bit us twice (an
  equivalence test, then the val selector).
- **`canonicalize_episode` is not idempotent** — always tensorize from *raw* episodes.
  Double canonicalization silently changes the object→tag binding and skewed every cached
  comparison number before 2026-07-26.
- **Selection metrics must not be censored below the tail that separates models.** A val
  FP censored at 30 attempts rated v2.2 and v3 equal (11.12 vs 11.40) while they differed
  by 4 FP uncensored on test; uncensoring it (G6b) moved v3 from *significantly worse* than
  v2.2 to indistinguishable. The tell is **dynamic range, not jitter** — the censored
  curves were stable and picked sensible mid-training epochs, they just spanned ≈6 FP where
  the uncensored ones span ≈15. Stable curves are not evidence of a good selector.
- **DD2D generation is `PYTHONHASHSEED`-dependent**, so no collection is reproducible
  across processes; expect a fresh sample on any re-collection.

## Documentation discipline — keep the living docs alive

The living docs are the project's research memory: code records *what*, the
docs record *why*. Commit messages and checkpoint dirs are not a research
log. After any change that is more than mechanical, decide — **before
committing** — which of these needs an entry, and ship the entry **in the
same commit** as the change:

| The work... | Update | Format |
|---|---|---|
| produced any run/EDA/probe/ablation numbers — **including failed and negative runs** | `docs/notebook/` | dated What / Result / Takeaway-next entry (format in its README) |
| chose between alternatives with lasting consequences, killed an approach, or changed a convention / invariant / metric / protocol | `docs/decisions/` | ADR: context → decision → consequences, newest first |
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

### Mechanics — both logs are chaptered by era

Both logs are split into era chapters under `docs/decisions/` and
`docs/notebook/` (same boundaries in both), with a **generated `README.md`**
per log. `docs/decisions/README.md` is what this file `@`-imports, so the
indexes and the standing-invariants table are always in context while the
narratives are read on demand.

```bash
python experiments/spectre/decisions_index.py new --log decisions \
    --title "..." --tracks method,evaluation   # scaffold into the open chapter
python experiments/spectre/decisions_index.py index                # regenerate READMEs
python experiments/spectre/decisions_index.py check                # enforced by pytest
```

Four rules, all enforced by `tests/approaches/spectre/test_doclog.py`:

- **Historical entries are append-only.** To change what an old entry says, add
  a new one, set `supersedes` on it and `superseded_by` + a `status` + a banner
  on the old one. `check` compares every pre-split entry against the frozen
  monolith in `docs/archive/` and fails on a silent edit. It also enforces that
  `supersedes` / `superseded_by` point at each other.
- **Status is machine-readable.** Each entry carries a fenced metadata strip
  (`<!--strip-->`) with id, status, tracks and cross-references, so a chunk
  retrieved out of context still says whether it is quoting a retracted number.
  The README's **do-not-quote** table lists every one of them.
- **Cite by entry id**, e.g.
  `decisions/05-v3-migration.md#2026-07-26-selection-metric-never-censored`.
  Ids are permanent. Older `` `decisions.md` <date> `` citations survive — the
  stubs still resolve and each README carries a legacy date→entry table — but
  dates collide (2026-07-19 has six entries), so new citations use the id.
- **Close a chapter** at a named phase boundary, or when `check` warns that the
  open one passed ~650 lines / 12 entries; add it to `_ERAS` in `doclog.py`.

`autorun_decisions.md` and `autonomous_stickbutton_session.md` stay as
standalone session narratives — they were judged without a human in the loop, so
they are *unratified*. Promote one to an ADR only if it changed a convention,
architecture or invariant **and** no ADR states that decision; promotion means
writing the ADR with `ratifies:` set and adding a forward pointer, never moving
the text. Four were promoted on 2026-07-29 (A3, A8+A11, A10, A16).
