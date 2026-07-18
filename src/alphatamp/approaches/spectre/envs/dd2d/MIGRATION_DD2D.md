# DD2D migration guide

This directory received the **DD2D** (Drawer Decluttering in 2D) TAMP environment,
migrated out of the `envsearch` research repo by `scripts/migrate_dd2d.py`. DD2D is
the plan-feasibility stress-test environment; everything the PIGINet data pipeline
needs to run on it is here, so a competing method (e.g. **SPECTRE**) can be trained
and evaluated on the same environment and record format.

- PIGINet baseline: **included (`piginet/` + its tests)**
- Generated data: **included (`data/dd2d/`, `out_dd2d/` under this package)**
- Import root: **`alphatamp.approaches.spectre.envs.dd2d`**

---

## 1. What DD2D is

A top-down household **drawer** holds 9-14 rotated parametric polygons including a
**target** that starts ungraspable. The robot stages a **subset of blocker items**
onto a wall-less **buffer** to clear the target, then **retrieves** it. The hard
decision is *which subset* -- a joint continuous **packing** feasibility problem.

Pipeline (per problem): generate an instance (`dd2d/problem.py`) -> enumerate
diverse task-plan skeletons with a FORBID-SEARCH / top-k planner (`planning.py` +
`dd2d/planning.py`) -> refine each skeleton geometrically (`dd2d/refine.py`) ->
emit a `PIGINetExample` record (`record.py` + `dd2d/record_ext.py`). The difficulty
is *measured, not installed*. Full design: `docs/dd2d.md`, `docs/dd2d_spec.md`.

## 2. Directory layout produced

```
dd2d/   # this package dir
  __init__.py     # TRIMMED: no clutter/stacking eager imports
  skeleton.py scene.py problem.py refine.py record.py rendering.py planning.py
  geometry/       # backends load lazily (pybullet only if you render 3D)
  dd2d/           # the environment (generators, planner, refiner, labeler, render)
  piginet/        # PIGINet baseline (only if included)
  domain/drawer_declutter.pddl   # DD2D operators/predicates
  domain/blocksworld.pddl        # planning.py default-path constant (DD2D overrides it)
  tests/          # test_dd2d*.py (+ test_piginet_*.py if included)
  requirements-blocks.txt requirements-piginet.txt
  docs/                    # dd2d*.md, piginet_dd2d_plan.md, piginet_record_schema.md
  scripts/check_mps.py     # MPS gate referenced by requirements-piginet.txt
  data/dd2d/  out_dd2d/    # generated datasets/outputs (only if included)
  MIGRATION_DD2D.md        # this file
```

### What was intentionally excluded (and why)

Not copied -- these belong to *other* environments / stacks in `envsearch` and DD2D
does not depend on them: `clutter.py`, `stacking.py`, `sweep.py`, `anytime.py`, the
root `collect.py`/`demo.py`, the 3D Panda refiners (`refine_backtracking.py`,
`refine_pybullet.py`, `refiners.py`, `trajectory_sampler.py`), `video.py`, the
`e1/` subpackage, `domain/capacitated_loading.pddl`, all sorting/clutter/stacking/e1
tests, and the sibling repos (`kitchen-worlds/`, `policy-guided-lazy-tamp/`,
`pyperplan/`, `papers/`, `della_md/`).

## 3. Package / import contract

Import root: **`alphatamp.approaches.spectre.envs.dd2d`**. Imports were **already rewritten** from `blocks_tamp` to `alphatamp.approaches.spectre.envs.dd2d` during migration (the shared layer used relative imports and was unchanged; `piginet/` uses relative imports too). Nothing further to do.

DD2D imports this shared layer (relative to the import root): `skeleton, scene,
problem, refine, record, rendering, planning, geometry`. The only file changed
during migration is the package `__init__.py` (trimmed to drop the `.clutter` /
`.stacking` eager imports; `problem.py`'s `make_problem` still dispatches to any
available env lazily).

Run modules with the package importable (installed, or its parent on `PYTHONPATH`):

```shell
python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.demo --lambda 0.6 --seed 0 --num-problems 2 --crowd 10
```

## 4. The SPECTRE <-> PIGINet interface

Both methods consume the **`PIGINetExample`** record (objects, init/goal literals,
task-plan skeleton, image refs, feasibility label). Schema + `fastamp` contract:
`docs/piginet_record_schema.md`; construction in `record.py` (`build_example`,
`build_image_refs`) and the DD2D extension `dd2d/record_ext.py`
(`build_dd2d_example`, crop writer). Records are produced in bulk by
`dd2d/collect.py`.

**Label caveat (read before reporting numbers):** DD2D uses a Day-1 labeler
(`dd2d/label.py`) -- a positive accessible-packing certificate + a sound area bound;
non-area-proven negatives are marked **marginal**, never hard-infeasible. Negatives
are provisional until the arrangement-complete negative certificate lands, so **no
label-dependent research numbers** until then. See `docs/dd2d.md`.

PIGINet baseline entry points (if included): `piginet/train.py`, `piginet/eval.py`
(frozen CLIP ViT-B/32 encoders; metrics via scikit-learn). Note: `record.py`'s
`generator` provenance string and one help string in `dd2d/demo.py` still literally
say `blocks_tamp...` -- cosmetic only, they are not imports.

## 5. Filepaths to fix / wire

- **PDDL domain seam:** `DD2DProblem.domain_pddl_path` resolves file-relative
  (`os.path.dirname(dirname(__file__))/domain/drawer_declutter.pddl`), so it keeps
  working as long as `domain/` stays a sibling of `dd2d/`. No edit needed.
- **Output roots:** DD2D writes under `out_dd2d/`, `data/dd2d/`, `out/dd2d/` by
  default; all are set by flags -- `dd2d/collect.py --out-root`, the demo `--out` --
  so point them wherever this project keeps artifacts.
- **Packaging data:** editable installs read `.pddl` / data in place (fine). For a
  built wheel, add the `.pddl` files (and any shipped data) to `package_data` /
  `MANIFEST.in`, since `domain/` has no `__init__.py`.
- **Hunt for stray absolute paths** before running:
  ```shell
  grep -rnE "/Users/|/home/|out_dd2d|data/dd2d" dd2d
  ```

## 6. Dependencies

- `requirements-blocks.txt` -- the environment/data stack. **`shapely` is a HARD
  DD2D dependency** (world/shapes/grasps/enumerate/label/scene/render all use it),
  despite the file's comment attributing it to E1. Also `unified-planning` +
  `up-symk` + `pyperplan` (diverse planning), `numpy`, `matplotlib`,
  `imageio`(+`imageio-ffmpeg`), `marimo` (EDA notebooks). `pybullet` is **optional**
  -- only for `confirm_rendering` / 3D `get_backend`; DD2D's own `render.py` is
  matplotlib-based.
- `requirements-piginet.txt` -- the baseline (torch, torchvision, open_clip_torch,
  scikit-learn, pandas, wandb). MPS gate: `scripts/check_mps.py`.

## 7. Run & verify

```shell
# (install deps into your environment first; shapely + the planners are required)
python -c "import alphatamp.approaches.spectre.envs.dd2d, alphatamp.approaches.spectre.envs.dd2d.dd2d.planning, alphatamp.approaches.spectre.envs.dd2d.dd2d.refine, alphatamp.approaches.spectre.envs.dd2d.record"      # import cleanliness -- must succeed with clutter/stacking absent
python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.demo --lambda 0.6 --seed 0 --num-problems 2 --crowd 10         # end-to-end demo
python -m pytest tests/test_dd2d.py -q
```
