# kinder / StickButton2D — codebase map for SPECTRE

**Purpose.** SPECTRE's method comparison currently lives on one environment (DD2D). This
document maps the upstream substrate needed to add **StickButton2D** as a second, so a fresh
or compacted session can start work without re-deriving any of it.

**Status (2026-07-29).** Mapping complete. A geometry-aware plan generator (`envs/stickbutton2d/`)
is built and deployed; **b1/b2/b3/b5 are collectable, b10 is not** — see
[`autonomous_stickbutton_session.md`](autonomous_stickbutton_session.md) for the heuristic and
[`decisions.md` 2026-07-29](decisions/06-v3-performance.md#2026-07-29-stickbutton2d-heuristic-distance-term) for the ADR. §7 records the measurements, including negative results;
some of its b5 conclusions are marked superseded there. No full dataset has been collected yet.

Everything below refers to the installed packages under
`.venv/lib/python3.11/site-packages/` (abbreviated `$SP`). Versions: `kindergarden` 0.2.0
(imports as `kinder`), `bilevel_planning` 0.1.4, `kinder_bilevel_planning` 0.1.0,
`kinder_models` 0.1.0, `relational_structs` 0.0.1.

---

## 1. Package layout

`kinder` is the environment layer only. It contains **no** planning code and **no**
`create_*` factories — construction is via gym `entry_point` + `kwargs`.

| Path | What |
|---|---|
| `$SP/kinder/__init__.py` | registry, `register_all_environments()`, `make()` |
| `$SP/kinder/core.py` | `ObjectCentricKinDEREnv`, `ConstantObjectKinDEREnv`, `FinalConfigMeta` |
| `$SP/kinder/envs/utils.py` | shared 2D geometry: collision, rendering, multibody construction |
| `$SP/kinder/envs/kinematic2d/base_env.py` | `Kinematic2DRobotEnvConfig`, base kinematics + collision rejection |
| `$SP/kinder/envs/kinematic2d/object_types.py` | `CRVRobotType`, `RectangleType`, `CircleType` |
| `$SP/kinder/envs/kinematic2d/structs.py` | `ZOrder`, `Body2D`, `MultiBody2D`, `SE2Pose` |
| **`$SP/kinder/envs/kinematic2d/stickbutton2d.py`** | **the environment (379 lines)** |
| `$SP/kinder_bilevel_planning/env_models/kinematic2d/stickbutton2d.py` | predicates, operators, skills |
| `$SP/kinder_models/kinematic2d/envs/stickbutton2d/parameterized_skills.py` | the controllers |
| `$SP/bilevel_planning/` | planners, plan generators, refiners, trajectory samplers |

Note `kinder/envs/` and `kinder/envs/kinematic2d/` have **no `__init__.py`** (implicit
namespace packages).

### Variant registration — `kinder/__init__.py:162-178`

```python
num_buttons = [1, 2, 3, 5, 10]
for num_button in num_buttons:
    variant_id = f"kinder/StickButton2D-b{num_button}-v0"
    _register(id=variant_id,
              entry_point="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",
              kwargs={"num_buttons": num_button})
```

So the five ids are `kinder/StickButton2D-b{1,2,3,5,10}-v0`. The `b{N}` suffix is purely
`num_buttons`. Any other count works if registered manually —
`spectre/env_registry.py:172 stick_button_variants()` exists for that and is a no-op for the
five pre-registered counts.

**Default mismatch worth knowing:** `ObjectCentricStickButton2DEnv` defaults to
`num_buttons=2`; the registered `StickButton2DEnv` defaults to `3`.

---

## 2. The environment

`$SP/kinder/envs/kinematic2d/stickbutton2d.py`

| Symbol | Lines |
|---|---|
| `StickButton2DEnvConfig` | 29–104 |
| `ObjectCentricStickButton2DEnv` | 107–332 |
| `._sample_initial_state()` | 125–179 |
| `._create_constant_initial_state_dict()` | 181–215 |
| `.step()` — **the press logic** | 284–317 |
| `._get_reward_and_done()` | 319–332 |
| `StickButton2DEnv` (the registered gym class) | 335–379 |

### Two-layer pattern (`core.py`)

`ObjectCentricKinDEREnv` (obs = `ObjectCentricState`) is wrapped by
`ConstantObjectKinDEREnv`, which resets once at construction to get an exemplar state, fixes an
ordered object list, and vectorizes observations into a `Box`.

Gotcha: `ConstantObjectKinDEREnv.__init__` consumes `render_mode` and does **not** forward it,
so the inner env is always `rgb_array` and `env.render()` works even when the outer
`render_mode is None`.

### Objects

| name | type | in obs? |
|---|---|---|
| `robot` | `CRVRobotType` (9 features) | yes |
| `stick` | `RectangleType` (10 features) | yes |
| `button0..N-1` | `CircleType` (9 features) | yes |
| `table` | `RectangleType`, `z_order=FLOOR` | **no** — constant object |
| `left/right/top/bottom_wall` | `RectangleType`, `z_order=ALL` | **no** — constant object |

Observation is `Box(-inf, inf, (9 + 10 + 9*num_buttons,))` → b1=28, b3=46, b5=64, b10=109.
Object order is `["robot", "stick"] + sorted(button names)` (`stickbutton2d.py:347`).

Feature layouts:
- `robot`: `x, y, theta, base_radius, arm_joint, arm_length, vacuum, gripper_height, gripper_width`
- `stick`: `x, y, theta, static, color_r, color_g, color_b, z_order, width, height`
- `button`: `x, y, theta, static, color_r, color_g, color_b, z_order, radius`

Walls and the table live in `initial_constant_state` and are merged in only for collision
checking and rendering. Reach them via
`env.unwrapped._object_centric_env.initial_constant_state`.

`RectangleType` geometry: `(x, y)` is the **lower-left corner**, not the center.

There is **no "held" flag**. Grasping is implicit: `vacuum > 0.5` plus geometric intersection of
the suction rectangle with a movable object (`get_suctioned_objects`,
`kinematic2d/utils.py:185`). That is exactly what the `Grasped` predicate calls.

### Action space — `CRVRobotActionSpace`

`Box((5,))` = `[dx, dy, dtheta, darm, vac]`, bounds `±0.05, ±0.05, ±π/16, ±0.1, [0,1]`.
`vac` is **absolute** (0/1); the rest are deltas. `darm` is clipped into
`[base_radius, arm_length] = [0.1, 0.2]`.

### Motion model

`base_env.py:141-199`: integrate the deltas, snap suctioned objects, then **reject the entire
transition if any collision results**. Motion is kinematic-with-rejection — there is no physics
and no motion planning. A blocked straight-line path simply stops making progress.

`truncated` is always `False` — no horizon. Wrap with `TimeLimit` if you need one.

### How a button gets pressed — `stickbutton2d.py:292-311`

```python
for button in self._current_state.get_objects(CircleType):
    if state_2d_has_collision(full_state, {button}, {robot, stick},
                              self._static_object_body_cache, ignore_z_orders=True):
        newly_pressed_buttons.add(button)
```

Three consequences that matter:

1. **"Pressed" is stored only as the button's colour** — `(0.9,0,0)` → `(0.0,0.9,0)`. The
   `color_r/g/b` slots of each button in the observation *are* the press flags.
2. **`ignore_z_orders=True`**, so any robot body (base, gripper, arm, and the suction rect when
   the vacuum is on) or the stick that geometrically intersects a button presses it — including
   incidentally, in passing.
3. **Presses are irreversible.** Nothing ever un-presses.

Termination: `reward = 0.0` iff every button is pressed, else `-1.0`.

---

## 3. Geometry — the reachability mechanism

All values verified at runtime from `StickButton2DEnvConfig()`.

| Quantity | Value |
|---|---|
| world | `x ∈ [0, 3.5]`, `y ∈ [0, 2.5]` |
| **table** | `pose=(0, 1.25)`, `shape=(3.5, 1.25)` → occupies `y ∈ [1.25, 2.5]`, `z_order=FLOOR` |
| floor | `y ∈ [0, 1.25]` |
| robot base radius | `0.1` |
| robot arm length | `0.2` (min `arm_joint` = `0.1`) |
| gripper h × w | `0.07 × 0.01` |
| robot init pose | `x ∈ [0.3, 3.2]`, `y ∈ [0.3, 0.95]` — **always starts on the floor** |
| stick shape | `0.05 × 1.25`, init `y ∈ [0.625, 1.125]`, `θ=0` — always vertical, straddling the table edge |
| button radius | `0.05` |
| **button init bounds** | `x ∈ [0.05, 3.45]`, `y ∈ [0.05, 2.45]` — **anywhere, floor or table** |

### `ZOrder` is the entire reachability story — `kinematic2d/structs.py:16-31`

```python
FLOOR = 1; SURFACE = 2; ALL = 100; NONE = 101
def z_orders_may_collide(z0, z1):
    if ZOrder.NONE in (z0, z1): return False
    if ZOrder.ALL  in (z0, z1): return True
    return z0 == z1
```

- Table = `FLOOR`, robot **base** = `ALL` ⇒ the base collides with the table and cannot enter it.
- Robot **gripper/arm** = `SURFACE` vs table `FLOOR` ⇒ no collision ⇒ the arm sweeps *over* the table.
- Stick = `SURFACE` ⇒ also passes over the table; blocked only by the `ALL` walls.
- Buttons = `NONE` ⇒ **buttons never block anything's motion.**

### The derived reach limit — the one number the symbolic model is missing

```
base-centre max y = table_y0 − base_radius        = 1.25 − 0.1   = 1.15
max pressable button y = 1.15 + arm_length + gripper_width/2 + button_radius
                       = 1.15 + 0.2 + 0.005 + 0.05              = 1.405
```

(`1.42` with the vacuum on, since the suction body extends a further `0.015`.)

**A button with `y ≳ 1.405` can only be pressed with the stick.** Buttons are sampled uniformly
over `y ∈ [0.05, 2.45]`, so **≈43% of buttons need the stick** in expectation.

The stick, held at the gripper, reaches `≈ 1.15 + 0.2 + 0.01 + 1.25 ≈ 2.61 > 2.5`, i.e. it
covers the whole table.

### Collision skips static-static pairs — `envs/utils.py:98-137`

```python
if obj1 == obj2 or (obj1_static and obj2_static):
    continue
```

So at generation time **buttons may overlap each other, the walls, and the table**; they only
avoid the robot and the stick. The robot lacks a `static` feature and is never treated as static.

---

## 4. The symbolic model

`$SP/kinder_bilevel_planning/env_models/kinematic2d/stickbutton2d.py`,
`create_bilevel_planning_models(observation_space, action_space, num_buttons)` at L38.

Reached from SPECTRE by `model_name: "stickbutton2d"` — the dispatcher
(`env_models/__init__.py:13-55`) does a **filesystem lookup**, not an import registry.

### Predicates (L69-82)

`Grasped(robot, stick)`, `HandEmpty(robot)`, `Pressed(button)`,
`RobotAboveButton(robot, button)`, `StickAboveButton(stick, button)`, and the **0-arity**
`AboveNoButton()`.

Abstractor semantics (L85-147): `Pressed` from the button's RGB (`atol=1e-3`);
`RobotAboveButton` iff **any** robot geom intersects the button; `AboveNoButton` iff neither the
robot nor the stick is above **any** button.

### Goal (L150-157) — static, independent of the state

```python
{GroundAtom(Pressed, [Object(f"button{i}", CircleType)]) for i in range(num_buttons)}
```

**This is load-bearing for feasibility:** the goal demands *all* N buttons pressed, so a single
unpressable button makes *every* skeleton infeasible.

### Operators (L165-266) — seven

`RobotPressButtonFromNothing(robot, button)`, `RobotPressButtonFromButton(robot, button, from_button)`,
`PickStickFromNothing(robot, stick)`, `PickStickFromButton(robot, stick, from_button)`,
`StickPressButtonFromNothing(robot, stick, button)`,
`StickPressButtonFromButton(robot, stick, button, from_button)`, `PlaceStick(robot, stick)`.

Modelling facts that matter:

1. **`Pressed` is monotone** — no operator deletes it. `#unpressed` is therefore an admissible
   heuristic.
2. **No geometry.** `RobotPressButton*` is applicable to *any* button, including ones physically
   out of reach. This is the gap a geometry-aware heuristic must close.
3. **`?button == ?from_button` self-loops are legal groundings** —
   `all_ground_operators` uses `get_object_combinations`, which allows repeats.
4. `StickPressButton*` **never mentions `RobotAboveButton`**, so the model assumes the robot-body
   relation is untouched while stick-pressing.
5. `AboveNoButton` acts as a global flag; `PlaceStick` and `PickStickFromButton` are the only
   things that re-add it.

### Skills / controllers

`$SP/kinder_models/kinematic2d/envs/stickbutton2d/parameterized_skills.py:289-421`. Paired 1:1
with the operators (`LiftedSkill.__post_init__` asserts the variable tuples match exactly).

**Parameter spaces — the single most important budgeting fact:**

| controller | params |
|---|---|
| `PickStick*` | `(grasp_ratio ∈ [0,1], arm_length ∈ [0.1,0.2])` — **genuinely sampled** |
| `RobotPressButton*` | `Box([0.0],[0.0])` — degenerate |
| `StickPressButton*` | `Box([0.0],[0.0])` — degenerate |
| `PlaceStick` | `Box([0.0],[0.0])` — degenerate |

⇒ **Resampling a failed press/place step is a pure no-op.** `num_sampling_attempts_per_step > 1`
buys nothing except at `PickStick` steps, where backtracking has its only real freedom.

Controllers are open-loop waypoint followers (`kinder_models/kinematic2d/utils.py:15-128`): the
plan is computed **once** at the first `step()` and never replanned; `observe()` only stores
state. There is no obstacle avoidance.

> ⚠️ **Do not read that last sentence as a diagnosis.** It is true of the controllers and
> irrelevant to why refinement fails: buttons are `ZOrder.NONE` and block nothing, and the
> only barrier is the table, which a correct plan never drives at. Reasoning from this
> property instead of from failure data produced a wrong conclusion once already — the actual
> causes are in §7.

---

## 5. The planning substrate (`bilevel_planning`)

### `SesameModels` — `structs.py:229-255`

`observation_space, state_space, action_space, transition_fn, types, predicates,
observation_to_state, state_abstractor, goal_deriver, skills`, plus optional
`ground_operators`. `.operators` is derived as `{s.operator for s in skills}`.

`transition_fn` for this env is a full `sim.reset(options={"init_state": state}); sim.step(u)`
per action — **it dominates all cost.**

### Abstract plan generators

`abstract_plan_generators/heuristic_search_plan_generator.py`

- `HeuristicSearchAbstractPlanGenerator(heuristic_factory, abstract_successor_function, seed)`
  (L67-148) — A* with unit action cost, priority `g + h`, RNG used only for tie-breaking. It is
  a **generator**: each goal node found is `yield`ed, goal nodes are not expanded, abstract
  states may be revisited (deliberately, to produce multiple skeletons), and dedup is on the
  action-plan tuple. The idiom everywhere is `itertools.islice(gen(...), K_max)`.
- `RelationalHeuristicSearchAbstractPlanGenerator(types, predicates, operators, heuristic_name, seed)`
  (L151-202) — the relational wrapper used by `spectre/collect.py`.

> ⚠️ **`heuristic_name` is accepted, stored, and then never used.** Line 198 hardcodes
> `create_pyperplan_heuristic("hff", ...)`. Passing `"hadd"`/`"lmcut"` has no effect. **The clean
> extension point for a domain-specific heuristic is to subclass
> `HeuristicSearchAbstractPlanGenerator` directly** and supply
> `heuristic_factory: Callable[[AbstractState, Goal], Callable[[AbstractState], float]]`, with a
> `RelationalAbstractSuccessorGenerator(operators)` as the successor function.

### Refiner — `refiners/backtracking_refiner.py:22-108`

`BacktrackingRefiner(trajectory_sampler, num_sampling_attempts_per_step, seed)`; `timeout` is
passed per call. It is the **only** refiner shipped. Depth-first over plan indices, retrying up
to `num_sampling_attempts_per_step` samples per step.

Two limitations to design around:

- **It returns `Plan | None` and nothing else** — no stuck-step index, no way to distinguish
  timeout from exhaustion. `spectre/collect.py:290` sets `stuck_step_index = None` on the kinder
  path for exactly this reason. Per-step attribution requires a subclass.
- **The timeout is only checked at the top of `_refine_from_step`**, and the elapsed budget is
  only decremented after a *successful* sample. A step that fails all its attempts runs the whole
  loop with no timeout check, so wall-clock overshoots the nominal budget substantially.

### Trajectory sampler — `trajectory_samplers/parameterized_controller_sampler.py`

`ParameterizedControllerTrajectorySampler(controller_generator, transition_function,
state_abstractor, max_trajectory_steps)`. Runs the controller to termination, then:

```python
if final_abstract_state == ns:      # line 89
    return x_traj, u_traj
raise TrajectorySamplingFailure()
```

**Acceptance is exact abstract-state equality**, not "add-effects achieved". Combined with §2's
incidental presses and §4.4's unmodelled `RobotAboveButton`, symbolically valid skeletons can be
rejected for reasons invisible in the PDDL. (See §7 — we tested relaxing this and it was *not*
the binding constraint.)

### Planners

`SesamePlanner(abstract_plan_generator, trajectory_sampler, max_abstract_plans,
num_sampling_attempts_per_step, ...)` constructs its `BacktrackingRefiner` internally and
**returns on first success**. `sesame.run_sesame(...)` is the convenience wrapper
(`samples_per_step → num_sampling_attempts_per_step`, `max_skill_horizon → max_trajectory_steps`).

SPECTRE does **not** use these: collection is deliberately non-short-circuiting.

### Determinism

All RNGs are `np.random.default_rng(seed)` built in `__init__`. Hashing is session-stable
(`prpl_utils.consistent_hash` = SHA-256 over `repr`), so successor enumeration order is stable
across processes **without** `PYTHONHASHSEED` — unlike DD2D. A fresh `BacktrackingRefiner` per
skeleton (what `collect.py` does) is what makes per-skeleton outcomes independently reproducible.

`BilevelPlanningGraph` inserts do linear `in` scans over lists — O(n²) overall. Sharing one graph
across a 200-skeleton pool is a real cost.

---

## 6. How SPECTRE wires this up today

`spectre/collect.py::collect_episode` already runs kinder envs end-to-end:

1. `kinder.make(cfg.env_id)`, `env.reset(seed=problem_id)`
2. `create_bilevel_planning_models(cfg.model_name, obs_space, act_space, **cfg.model_kwargs)`
3. `x0 = observation_to_state(obs)`, `s0 = state_abstractor(x0)`, `goal = goal_deriver(x0)`
4. `RelationalHeuristicSearchAbstractPlanGenerator(...)` → `islice(..., K_max)`
5. one `ParameterizedControllerTrajectorySampler` reused across skeletons
6. per skeleton, a fresh `BacktrackingRefiner` with a blake2b-derived seed; **every** skeleton is
   refined regardless of earlier successes
7. → `EpisodeRecord` gzip-pickled to
   `data/spectre/raw/<env_variant>/<split>/episodes/ep_<id>.pkl.gz`

`conf/env/stickbutton2d_b5.yaml` and `submit_spectre_stickbutton2d_b5.sh` already exist and are
correct.

**What this path does *not* fill:** `scene_geometry=None` and `refiner_metadata={}` for kinder
envs, so the v3 `FailureRecord` pathway is inert and `domain.spec_for("stickbutton2d_*")` falls
back to `EMPTY_SPEC`. Also note `dataset.py` drops episodes with `num_success == 0` or
`num_skeletons < 2`.

---

## 7. Measured findings (2026-07-28)

Measured directly, not inferred. **Includes negative results** — do not re-run these.

### Stock pipeline is far too sparse to train on

`RelationalHeuristicSearchAbstractPlanGenerator` (hff) + `BacktrackingRefiner`,
`num_sampling_attempts_per_step=5`, `max_trajectory_steps=200`, `refinement_timeout_s=20`:

| variant | pool | skeletons refined |
|---|---|---|
| b1 | 12 × 3 seeds | 15/36 (42%) |
| b2 | 12 × 3 seeds | 9/36 (25%) |
| b3 | 12 × 3 seeds | 0/36 |
| b5 | **200** | **0 successes**, 379 s |
| b10 | 40 | **0 successes** |

Refinement fails *fast* (≈0.2–0.8 s per skeleton), so this is not a timeout problem.

### Cause 1 — the generator is geometry-blind

In b3/seed0 the buttons sit at `y = 1.80, 2.29, 0.057`: two are deep table buttons requiring
the stick (§3), but hff ranks bare-robot `RobotPressButton` plans on them first because they are
symbolically shortest. Per-operator mismatch counts at b3: `RobotPressButtonFromButton` 1010,
`RobotPressButtonFromNothing` 20.

**A geometry-aware heuristic helps where reach binds** (prototype measured):

| case | hff | geometry-aware |
|---|---|---|
| b3 seed0 (2 stick-buttons) | first success @ idx 29 | **@ idx 16** |
| b3 seed1 (0 stick-buttons) | @ idx 14 | @ idx 14 — correctly reduces to hff |
| b5, b10 | none in 40–200 | still none |

### Cause 2 — per-button achievability compounds

Because the goal is *all* N buttons (§4), one unpressable button voids every skeleton, so
episode feasibility falls roughly as `Π q_i` in the per-button achievable rate. This — not
ordering — is what kills b5/b10, and no heuristic can fix it.

The per-button probe (`diagnostics.py`, 20 problems per variant) under stock exact
acceptance:

| variant | predicted-solvable | mean s/problem |
|---|---|---|
| b1 | 100% | 0.3 |
| b2 | 55% | 1.9 |
| b3 | 35% | 3.3 |
| b5 | **0%** | 9.7 |
| b10 | **0%** | 27.9 |

### Stick pickup failure is deterministic per scene, not sampling luck

A frequent blocker is `missing=('(Grasped robot stick)',)` — `PickStickFromNothing` itself
fails, which kills *every* stick-dependent button in that scene at once. `PickStick` is the
one controller with genuinely sampled parameters (§4), so it is the only place
`num_sampling_attempts_per_step` can possibly help. It does not: over 10 b5 scenes, raising it
5 → 25 → 100 left the rate at **7/10 with the identical three scenes failing every time**
(mean cost 0.4 s → 1.5 s → 5.3 s). Some scenes simply place the stick where the robot cannot
grasp it. Raising sampling budgets is not a lever here; **filtering problems is.**

### Relaxing the acceptance test: helps per-button, does **not** survive chaining

Replacing `final_abstract_state == ns` with `ns.atoms ⊆ final_abstract_state.atoms`
(`sampler.py`, `acceptance="superset"`). Sound for goal achievement: all preconditions are
positive and `Pressed` is never deleted, so if every step achieves at least its planned atoms
then the final state ⊇ final planned ⊇ goal.

It transforms the *per-button* numbers, because `extra_atoms` (incidental presses,
multi-button overlap) is the largest single blocker under exact acceptance — 94 of b10's
blockers, 34 of b5's:

| variant | exact | superset |
|---|---|---|
| b2 | 55% | 90% |
| b3 | 35% | 85% |
| b5 | 0% | **75%** |
| b10 | 0% | **55%** |

**But it buys nothing end-to-end.** On real skeletons (full mode, geometry-aware generator,
`k_max=60`, 8 problems per variant): **b3 is 8/8 with or without it, and b5 is 0/8 either
way.** Once an incidental press makes the world diverge from the symbolic plan, later steps
are checked against a plan that no longer describes reality.

**Stock kinder acceptance semantics are therefore kept**: this relaxation is a deviation that
does not pay for itself. `acceptance="superset"` stays available and measurable, off by
default.

### Ground truth: b1–b3 are collectable, b5/b10 are not

`full` mode, geometry-aware generator, stock exact acceptance, `k_max=60`, 8 problems per
variant, every candidate refined (non-short-circuiting):

| variant | has ≥1 success | median first-success idx | mean #success / 60 | mean s/problem |
|---|---|---|---|---|
| b1 | **8/8 (100%)** | 0 | 24.2 | 465 |
| b2 | **8/8 (100%)** | 3 | 9.1 | 313 |
| b3 | **8/8 (100%)** | 14 | 2.8 | 280 |
| b5 | **0/8** | — | 0 | 133 |
| b10 | 0 (probe) | — | 0 | — |

> **Superseded for b5 on 2026-07-29** — with the geometry-aware heuristic (distance-to-nearest
> + reach grounding) b5 reaches **15/20 (75%)** at a 200-candidate budget, and b3 reaches
> 20/20 with its first success at index 2–10. b10 is unchanged at 0/20.

b1–b3 clear the 80–90% bar outright. Positives thin out fast with button count
(24.2 → 9.1 → 2.8 per 60 candidates), which is what makes b3 the interesting ranking problem
and b5 the cliff.

### The probe is a diagnostic, not a bound — in either direction

Worth stating plainly because it was initially recorded as an upper bound and that is wrong.
Against ground truth it errs **both** ways:

- **Under**-estimates: probe said b2 55% / b3 35%; the true rate is 100% for both. Probing
  from `x0` tries one route per button, whereas a real skeleton can reach the same button
  from a different predecessor (via `RobotPressButtonFromButton`, a different approach path)
  or press it incidentally in passing.
- **Over**-estimates: under `superset` it said b5 75%; the true rate is 0%. A real skeleton
  must *chain* presses, and each extra step is another chance to fail.

Use it for **failure attribution** (out-of-reach vs stuck controller vs extra atoms), which
is what produced the findings above. Use `full` mode to decide whether a variant is
collectable.

### Why b5/b10 fail: two blockers that mask each other (and it is NOT obstacle avoidance)

Attributing the b5 failures step by step (rather than inferring from controller properties)
gives a different answer than "the controllers can't avoid obstacles". **Nothing in this
environment needs avoiding** — buttons are `ZOrder.NONE` and never block motion, and the only
real barrier is the table, which a *correct* plan never drives at because table buttons are
the stick's job.

Under the unpruned pool, **102 of 120 skeletons fail at step 0** — the first action, so this
is not length-compounding either. Two independent causes:

1. **Out-of-reach plans → fail with *missing* atoms.** The heuristic's `+1` surcharge is a
   constant; it makes stick plans competitive but does not make an impossible action look
   bad, because pressing a table button still reduces `|unpressed|` by one and so looks
   optimal to A*. Fix: do not *ground* `RobotPressButton*(robot, b)` for `b ∈ needs_stick` at
   all. Reach as an applicability restriction, not a cost hint. (b5 grounding 67 → 49–61 ops.)
2. **Incidental presses → fail with *extra* atoms.** With grounding pruned, failures become
   `missing=()`, `extra=('(Pressed button2)','(Pressed button4)')` — the robot does exactly
   what it was asked and *also* presses buttons it drives over. In b5/seed0 button2 and
   button3 are 0.17 apart (buttons may overlap: generation skips static-static collisions);
   in seed2, merely fetching the stick sweeps button0.

**Pressing a button twice is free, and "exactly once" is not the requirement.** `Pressed`
comes from the button's colour and is never deleted, so re-pressing a green button is not
even observable in the abstract state. What breaks is that the world runs **ahead of
schedule**: the exact-equality check at `parameterized_controller_sampler.py:89` demands the
state after each step equal the planned state, and an early press of a *future* target
violates that while the final state would have been correct. Measured over b5 seeds 0–3:

```
pressed_early_but_planned_later   41
pressed_never_planned_later        0     <- not one stray press
extra_non_Pressed                 29     (RobotAboveButton / AboveNoButton)
```

Every single incidental press was of a button the remaining plan suffix still intended to
press. The trajectory does the right work in the wrong order.

**Each fix alone measures as useless, because the other masks it.** Pruning alone: still 0 at
b5 (incidental presses). Superset alone: still 0 at b5 (out-of-reach plans). Testing them
separately is how this was first — wrongly — written off as needing controller work.

**Together they help but do not rescue the hard variants:**

| | stock | prune + superset |
|---|---|---|
| b5 | 0/4 seeds | **2/4** (seed0: 2 successes, seed3: 1) |
| b10 | 0/4 | **0/4** |

The residual is *missing*-atom failure that neither addresses — genuine controller failure,
chiefly `PickStick` on scenes where the stick is simply not graspable (the deterministic
3/10, above).

> **Superseded 2026-07-29.** b5 *is* collectable, via the abstract-planning route instead:
> a distance-to-nearest-button term in the heuristic plus reach-based grounding takes it to
> **15/20 (75%)** with the stock sampler, no acceptance relaxation. b10 remains 0/20 and is
> dropped. See `autonomous_stickbutton_session.md` and [`decisions.md` 2026-07-29](decisions/06-v3-performance.md#2026-07-29-stickbutton2d-heuristic-distance-term). The
> prune+superset numbers below are retained as the record of a path not taken.

### Cost anchor for a real collection

At `k_max=60` a b3 problem costs ~280 s. `K_max=200` is ~3× that, so 400/100/100 = 600
problems lands near **6–8 h at ~30-way parallelism**. Budget accordingly; the collector
refines every candidate by design.

---

## 8. Gotcha checklist

- `heuristic_name` on `RelationalHeuristicSearchAbstractPlanGenerator` is **ignored**; `hff` is hardcoded.
- `BacktrackingRefiner` reports `Plan | None` only — no stuck step, no timeout-vs-exhaustion.
- Its timeout is checked only at step entry ⇒ real wall-clock overshoots the budget.
- Only `PickStick*` has non-degenerate sampled parameters; retries elsewhere are no-ops.
- `TrajectorySamplingFailure`, `TransitionFailure` and `AgentFailure` subclass **`BaseException`** —
  `except Exception` silently misses them (`collect.py` uses `except BaseException`).
- Sampler acceptance is **exact** abstract-state equality.
- `transition_fn` is a full sim reset + step per action; it dominates cost.
- `BilevelPlanningGraph` inserts are O(n) ⇒ quadratic across a large pool.
- Grounding admits `?button == ?from_button` self-loops.
- Buttons never block motion (`z_order=NONE`); the **table** blocks the robot base, the stick and
  arm pass over it.
- Buttons may overlap each other/walls/table at generation (static-static collisions are skipped).
- `ObjectCentricStickButton2DEnv` defaults to 2 buttons, `StickButton2DEnv` to 3.
- Unlike DD2D, generation here **is** reproducible across processes (`consistent_hash`).
