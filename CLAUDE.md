# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation

AlphaTAMP is a shared research monorepo for the PRPL lab covering multiple projects on "learning to accelerate TAMP" (task-and-motion planning). Each project lives under `src/alphatamp/approaches/` as a self-contained *approach* that plugs into a common bilevel-planning substrate from upstream lab packages.

**Per-project context lives in each approach's own directory** — look for a `CLAUDE.md` inside `src/alphatamp/approaches/<approach>/` before doing substantive work on that project. The currently active project is SPECTRE: see `src/alphatamp/approaches/spectre/CLAUDE.md` (its specs, experiment layout, and conventions live under that package's `docs/`).

Monorepo-wide must-read: `KEY_DEPENDENCIES.md` — the minimal upstream-package + substrate-file list that must be understood to touch this repo.

## Environment and commands

Python 3.11 is required. The project is managed with `uv`. A pre-built venv exists at `.venv`.

- **Activate venv (required for every shell):** `source .venv/bin/activate`
- Never call a global `python` — always use the venv's `python`/`pytest`.
- Install deps (rarely needed; venv is pre-populated): `uv pip install -e ".[develop]"`

CI and formatting:
- `./run_autoformat.sh` — runs `black .`, `docformatter -i -r .`, `isort .`. Line length 88.
- `./run_ci_checks.sh` — autoformat, then `mypy .` (strict equality, disallow untyped calls, warn unreachable), then `pytest . --pylint -m pylint`, then `pytest tests/`.
- Single test: `pytest tests/approaches/test_pure_planning_approach.py::test_name -xvs`
- Slow tests are skipped by default (`conftest.py` sets `markexpr = "not slow"`); opt in with `pytest tests/ -m slow` or `-m ""`.
- Some tests can emit videos: `pytest --make-videos ...` (custom flag from `tests/conftest.py`).

## High-level architecture

### The upstream substrate (do not modify — lives in installed packages)

All approaches consume a common substrate from the PRPL monorepo dependencies:
- `relational_structs` — `LiftedOperator`, `GroundOperator`, `GroundAtom`, `Predicate`, `Type`, `Object`, PDDL plumbing.
- `bilevel_planning` — `structs.py` (core types: `RelationalAbstractState`, `LiftedSkill`/`GroundSkill`, `SesameModels`, `ParameterizedController`, `PlanningProblem`, `Plan`), `pddl.py`, abstract plan generators, trajectory samplers.
- `kinder` / `kinder-bilevel-planning` / `kinder-models` — the five 2D environments (`ClutteredRetrieval2D`, `ClutteredStorage2D`, `Motion2D`, `Obstruction2D`, `StickButton2D`) and their env-models factory (`create_bilevel_planning_models`).

A `Skeleton` in this repo (`src/alphatamp/structs.py`) is `tuple[list[RelationalAbstractState], list[GroundOperator]]` — interleaved abstract-state sequence and ground-operator sequence produced by the symbolic planner.

### Two approach paradigms

Approaches inherit from one of two bases:
- `BaseApproach` (`approaches/base_approach.py`) — has access to a simulator; implements `train()` and `run_planning()` taking a `PlanningProblem`. Used by approaches like the oracle/pure-planner and the LLM-based cluttered_storage approaches.
- `SimulatorFreeBaseApproach` (`approaches/simulator_free_base_approach.py`) — `Agent[_O,_U]` that cannot call a transition model; drives learning loops via `step()`/`update()`/`reset_episode()`. Convert `SesameModels → SimulatorFreeSesameModels` via `sesame_models_to_sim_free`. This is the path used by `SimFreeParamPolicyApproach` and the full data-collection pipeline.

A simulator-free approach is wired together from pluggable pieces: an *abstract explorer* (`abstract_explorers/`), a *feasibility classifier learner* + *feasibility classifier* (`feasibility_classifier{_learners,s}/`), an *abstract action scorer* (`scorers/abstract_action_scorers/`), a *parameter scorer* (`scorers/parameter_scorers/`), and a *q-network* (`abstract_plan_classifiers/q_network.py`). `experiments/collect_data.py` is the canonical example of composing these.

### Experiment runners (Hydra)

Entrypoints under `experiments/` use Hydra with configs in `experiments/conf/`:
- `run_experiments.py` (config `conf/config.yaml`) — single run or multirun sweep; picks `env/{cluttered_storage,dyn_obstruction}.yaml` and `approach/{oracle,pure_planner,simfree_param_policy}.yaml`. Writes `results.csv`.
- `collect_data.py` (config `conf/collect_data_config.yaml`) — per-seed dataset collection for `SimFreeParamPolicyApproach`. Sweep seeds via Hydra multirun (`-m`), optionally with `hydra/launcher=joblib` or `slurm`.
- `bandit_test.py` / `bandit_ablation.slurm` — bandit experiments (legacy BOX/Attempt-1 work; archive data in `archive/`).

Hydra approach configs use `_target_: <fully.qualified.ClassName>` and are instantiated via `hydra.utils.instantiate(cfg.approach, env_models, seed)`.

Project-specific experiment trees live in subdirectories of `experiments/` and are self-contained (entry points + their own `conf/`): e.g. `experiments/spectre/`. The shared `experiments/conf/` tree is for the runners listed above — do not add project-specific configs to it.

## Repo conventions worth knowing

- `src/alphatamp/structs.py` re-exports the `Skeleton` / `FrozenSkeleton` type aliases used across approaches; prefer these over re-defining tuples locally.
- Approaches that need to run without a simulator import `sesame_models_to_sim_free`; do not duplicate that stripping logic.
- Notebooks under `experiments/**/*.ipynb` are for analysis against collected pickles/datasets (`tests/datasets/*.pkl`, `experiments/*.pkl`). Treat them as scratch — they are not part of CI.
- `lib/` vendors external JS libraries for visualizations; `archive/` holds old experiment results; `unit_test_videos/` is a scratch output dir.
- There is no `.cursor/`, `.cursorrules`, or `.github/copilot-instructions.md` in this repo at time of writing.
