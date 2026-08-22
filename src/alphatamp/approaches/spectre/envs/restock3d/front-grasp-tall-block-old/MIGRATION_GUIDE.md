# Migration Guide: Front-Grasp + Tall-Block Shelf3D

This folder packages a **front-grasp pick + translate-only place** capability for a **tall,
skinny block** in the KinDER kinematic3D `Shelf3D` task, ready to drop into **another repo
that imports the kinder packages** (`kindergarden`, `kinder_models`, and optionally
`kinder_bilevel_planning`) as dependencies.

**What it does:** the mobile manipulator drives to an upright block on the floor, grasps it
**from the front** with the fingers on the block's **left/right (±x) faces** (a diagonal
~45° approach), carries it to the shelf, and **inserts it into a shelf cell without rotating
it** — the same face that was down on the floor is down on the shelf. See
`demo_videos/front_shelf3d_{skills,planner}.mp4` for the target result.

---

## 1. Files in this folder

| File | Role |
|---|---|
| `front_grasp_skills.py` | The two controllers (`FrontGroundPickController`, `FrontGroundPlaceController`) + `create_front_lifted_controllers` + all tunable constants. **Self-contained** — needs no changes to `kinder_models`. |
| `shelf3d_front.py` | The bilevel-planning **env-model builder** `create_bilevel_planning_models(...)` (predicates, operators, abstractor, goal, transition) + `TALL_BLOCK_HALF_EXTENTS`. |
| `demo_front_shelf3d.py` | Produces two mp4 demos (direct controllers + SeSamE planner). |
| `test_shelf3d_front.py` | Regression test: pick→place reaches the goal and the block stays upright. |
| `kinder_models_hook.md` | **Optional** alternative (Option B): a small patch to `kinder_models` if you'd rather not use the self-contained pick. |
| `demo_videos/` | Reference result videos. |

---

## 2. Prerequisites

Your repo must have these installed (they are all published/git-installable KinDER/PRPL
packages):

- `kindergarden` (provides `import kinder`, the `KinematicShelf3D` env, `Shelf3DEnvConfig`,
  `ObjectCentricShelf3DEnv`, the object types).
- `kinder_models` (provides `GroundPickController`, `GroundPlaceController`,
  `BasePlaceController`, constants, `get_target_robot_pose_from_parameters`).
- `relational_structs`, `bilevel_planning`, `pybullet_helpers` (usually pulled in
  transitively by the above).
- `kinder_bilevel_planning` — **only** if you use the SeSamE planner path
  (`BilevelPlanningAgent`). If you don't, delete the planner path in the demo and use
  `bilevel_planning.sesame.run_sesame` directly; nothing else needs it.
- Python **≥3.10, <3.13**.
- `imageio` + `imageio-ffmpeg` (only for writing the mp4 demos).

> These files do **not** modify any kinder package (unless you choose Option B). They only
> subclass/compose the public classes.

---

## 3. Drop-in placement + import renames

Put the four `.py` files into a package in your repo, e.g. `your_pkg/front_grasp/`. The
files reference each other with **bare module imports** (`from front_grasp_skills import
...`) so they also run as-is from a single directory. For a real package, rename **only the
intra-package imports** to your package path — the `kinder_*` / `bilevel_planning` /
`pybullet_helpers` / `relational_structs` imports stay exactly as they are.

Lines to change (search for the `# --- Change ... ---` markers):

- `shelf3d_front.py`: `from front_grasp_skills import create_front_lifted_controllers`
  → `from your_pkg.front_grasp.front_grasp_skills import create_front_lifted_controllers`
- `demo_front_shelf3d.py` and `test_shelf3d_front.py`:
  `from shelf3d_front import TALL_BLOCK_HALF_EXTENTS, create_bilevel_planning_models`
  → `from your_pkg.front_grasp.shelf3d_front import ...`

---

## 4. Two important gotchas (the reason this isn't pure copy-paste)

### 4a. The pick controller is self-contained on purpose

In the original kinder-baselines repo, the front pick relied on a tiny hook added to
`kinder_models`'s `GroundPickController` (an overridable `self._grasp_transform` + an
extracted `_plan_grasp_approach()`). A repo that **imports** stock `kinder_models` doesn't
have that hook, so the shipped `FrontGroundPickController` here **overrides `step()`
entirely** and depends on nothing but the public base class. Nothing to do — it just works.

- If you *own/fork* `kinder_models` and prefer the smaller subclass, apply the patch in
  `kinder_models_hook.md` and swap in the minimal pick shown there.
- The **place** controller subclasses `GroundPlaceController` and only reuses inherited
  `BasePlaceController` helpers, so it never needed a patch.

### 4b. Call the env-model builder DIRECTLY (not the string dispatcher)

`kinder_bilevel_planning.env_models.create_bilevel_planning_models("shelf3d_front", ...)`
resolves the `env_name` string to a file **inside the installed `kinder_bilevel_planning`
package** — it will *not* find your local `shelf3d_front.py`. So always import and call the
builder directly:

```python
import kinder
from kinder.envs.kinematic3d.shelf3d import Shelf3DEnvConfig
from your_pkg.front_grasp.shelf3d_front import (
    create_bilevel_planning_models, TALL_BLOCK_HALF_EXTENTS,
)

kinder.register_all_environments()
config = Shelf3DEnvConfig(block_half_extents=TALL_BLOCK_HALF_EXTENTS)   # tall block
env = kinder.make("kinder/KinematicShelf3D-o1-v0", render_mode="rgb_array", config=config)
models = create_bilevel_planning_models(          # <-- direct call, no env_name string
    env.observation_space, env.action_space, num_objects=1, config=config,
)
```

> The **same** `config` must be passed to both `kinder.make(...)` and
> `create_bilevel_planning_models(...)`, so the executable env and the planner's internal
> sim agree on the tall block. (`kinder.make` forwards `config=` through to the env.)

---

## 5. Environment / build gotchas

- **IKFast compiles once.** The Kinova arm's analytic IK (IKFast) builds a small C++ module
  the first time it's used. If that build fails because it wants **static** `liblapack.a` /
  `libblas.a` and only the shared `.so` are installed, point it at a shim directory whose
  `.a` names symlink to the shared libs, e.g.:
  ```bash
  mkdir -p /tmp/libshim
  ln -sf /usr/lib/x86_64-linux-gnu/liblapack.so.3 /tmp/libshim/liblapack.a
  ln -sf /usr/lib/x86_64-linux-gnu/libblas.so.3   /tmp/libshim/libblas.a
  LAPACK_DIR=/tmp/libshim BLAS_DIR=/tmp/libshim python demo_front_shelf3d.py
  ```
  Once compiled, the module is cached in the installed `pybullet_helpers` package and no
  env vars are needed on subsequent runs.
- **Python 3.12** removed `distutils`; if the IKFast build errors on `import distutils`,
  `pip install setuptools` (it provides the shim).
- **mp4 output** needs `imageio-ffmpeg` (`pip install imageio-ffmpeg`). If you can't install
  it, write GIFs instead (`iio.mimsave("out.gif", frames, fps=..., loop=0)`).

---

## 6. Tunable knobs (in `front_grasp_skills.py`)

The grasp geometry is fully described by `FRONT_GRASP_TRANSFORM_TO_OBJECT` and a few bounds.
The derivation (worth understanding before you re-tune):

- **Approach axis = tool +z; finger-closing axis = tool +x** (from the Robotiq 2F-85 model).
- A **fully horizontal** grasp of a block on the floor is IK-infeasible for this arm (it
  can't point the wrist horizontal that low). The most-horizontal approach feasible at both
  the ground pick and the shelf is **~45°**, so the approach is pitched 45°.
- To grip the **left/right (±x) faces**, tool +x must map to world x — a pure **roll about
  the approach axis**, always reachable because `joint_7` (the wrist roll) is a `continuous`
  joint and the IKFast free joint.
- The result is a pure **Rx(−135°)** orientation `(-0.9238795, 0, 0, 0.3826834)` (x,y,z,w):
  tool +z → world `(0, +0.707, −0.707)` (down-forward toward the shelf), tool +x → world +x
  (fingers on the ±x faces). Translate-only place keeps the block's orientation fixed, so it
  ends up upright regardless of the gripper pitch/roll.

| Constant | Meaning / when to re-tune |
|---|---|
| `FRONT_GRASP_TRANSFORM_TO_OBJECT` | Grasp orientation (above) + object-frame offset. The `+0.057` z-offset grasps near the block **top** (at 45° the gripper can't dip below ~z=0.12 without the fingers hitting the floor). Raise/lower for a shorter/taller block. |
| `TALL_BLOCK_HALF_EXTENTS` (in `shelf3d_front.py`) | `(0.025, 0.025, 0.0635)` → 0.05×0.05×0.127 m. The 5 cm grasp width is well within the 8.5 cm 2F-85 stroke. |
| `TARGET_LAYER` | Which shelf board to place on (1 = lowest that satisfies the goal). |
| `FRONT_PICK_DISTANCE_BOUNDS`, `FRONT_PICK_ROT_BOUNDS` | Where the base parks to pick (facing +y toward the block). |
| `PLACE_STANDOFF`, `PLACE_BASE_DISTANCE` | Pre-place standoff (backs off along the approach axis so the block settles from above-front) and base standoff at the shelf. |

If you change the **block size** or **shelf geometry**, re-check: the grasp z-offset (so the
fingers clear the floor and the marker overlaps the block), and `TARGET_LAYER` /
`desired_object_z` (so the block rests within the 5 mm release tolerance of the board).

---

## 7. Run & verify

From your package directory (imports resolved), in the env with the kinder packages:

```bash
# both demo videos (seed 123):
python demo_front_shelf3d.py
#   -> front_shelf3d_skills.mp4  (controllers driven directly)
#   -> front_shelf3d_planner.mp4 (SeSamE planner)

# regression test:
pytest test_shelf3d_front.py
```

Expected: the task reaches the goal (`OnFixture(cube0, shelf)` + `HandEmpty(robot)`), the
block ends **upright** (final orientation == identity), and the fingers contact the block's
**left/right faces**. The test asserts the first two directly.

**Robustness:** the pick is reliable across seeds; a single place attempt succeeds ~3/5
(the pre-place motion plan occasionally fails for a given sample). The **SeSamE planner
resolves this automatically** by resampling, and the direct-controller demo retries the
whole episode with fresh samples — so end-to-end success is reliable.

---

## 8. Design rationale (one paragraph)

Top-down grasping is wrong for tall blocks going into a shelf: the wrist collides with the
shelf above the cell, and any grasp that rotates the block would lay it on its side. The
front grasp fixes both — it approaches diagonally (as horizontal as the arm can manage) and
the **place is translate-only** (it reuses the orientation-preserving EE pose the base place
controller already computes but normally discards), so the block keeps its orientation from
floor to shelf. The final refinement was rolling the gripper 90° about its approach axis so
the pads press flat on the block's left/right faces instead of pinching the top and front —
a grip that actually holds in the real world.
